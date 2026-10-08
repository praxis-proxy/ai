// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Anthropic Messages to Chat Completions-compatible request transformation.

use serde_json::{Map, Value, json};
use tracing::warn;

use super::config::LossyFeatureAllowlist;
use crate::hash::{self, Sha256};

// -----------------------------------------------------------------------------
// Request Transformation
// -----------------------------------------------------------------------------

/// Request fields whose effect the translated response could not report.
///
/// The Anthropic fields change behavior that the response would misreport:
/// `wire.rs` hardcodes `service_tier`, `container` and `inference_geo` to
/// null, and `mcp_servers` is a beta server-side feature
/// (`mcp-client-2025-04-04`).
///
/// The Chat Completions fields have no Anthropic counterpart, so no
/// conforming client sends them, but forwarding would let them reach the
/// backend, which then produces output the response translators discard:
/// only the first of `n` choices is translated, `logprobs` and
/// `top_logprobs` are dropped, `audio` and `modalities` output parts are
/// dropped, the deprecated `functions`/`function_call` shape is never read,
/// `web_search_options` annotations are dropped, and the top-level
/// `moderation` results are dropped. Rejecting is honest; forwarding would
/// bill the client for output it never sees. A value equal to the field's
/// documented default changes nothing and is dropped instead (see
/// [`is_default_value`]).
const UNREPRESENTABLE_FIELDS: [&str; 13] = [
    "service_tier",
    "container",
    "inference_geo",
    "mcp_servers",
    "n",
    "logprobs",
    "top_logprobs",
    "audio",
    "modalities",
    "functions",
    "function_call",
    "web_search_options",
    "moderation",
];

/// Anthropic Messages fields without a Chat Completions equivalent.
///
/// The translated response cannot carry thinking blocks, `context_management`
/// can edit them, and a top-level `cache_control` applies a cache marker Chat
/// Completions has no concept of. Non-null values are rejected before
/// conversion; null values have no effect and are removed. Allowlisted lossy
/// features ([`degrade_extended_thinking`], [`degrade_prompt_caching`]) strip
/// their markers first, so a value only reaches this rejection when it was not
/// degraded or was malformed.
const UNMAPPABLE_FIELDS: [&str; 3] = ["thinking", "context_management", "cache_control"];

/// Every `output_config` key in the Anthropic schema, including the beta
/// `task_budget`; the schema declares no others (`additionalProperties:
/// false`) and every one is nullable.
const OUTPUT_CONFIG_KEYS: [&str; 3] = ["effort", "format", "task_budget"];

/// Features the translator degraded instead of rejecting, one flag per
/// allowlisted [`LossyFeature`](super::config::LossyFeature).
///
/// A flag is set only when a recognized marker was actually removed, so the
/// operator signals report real degradation rather than mere opt-in.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct DegradedFeatures {
    /// A recognized `thinking` request or thinking-only `context_management`
    /// edit was removed.
    pub extended_thinking: bool,
    /// A valid `cache_control` marker was removed.
    pub prompt_caching: bool,
}

impl DegradedFeatures {
    /// Whether any feature was degraded.
    pub(crate) fn any(self) -> bool {
        self.extended_thinking || self.prompt_caching
    }
}

/// The translated request body plus the features that were degraded producing it.
#[derive(Debug)]
pub(crate) struct TransformOutput {
    /// The transformed Chat Completions-compatible request bytes.
    pub body: Vec<u8>,
    /// The features degraded while translating.
    pub degraded: DegradedFeatures,
}

/// Transform an Anthropic Messages request with strict fidelity: every
/// unrepresentable feature is rejected. Equivalent to
/// [`transform_request_degrading`] with an empty allowlist.
///
/// A test-only convenience: the filter always calls
/// [`transform_request_degrading`] with the configured allowlist, and the
/// empty-allowlist path is exercised through the strict tests and
/// [`empty_allowlist_matches_strict_wrapper`].
#[cfg(test)]
pub(crate) fn transform_request(value: Value) -> Result<Vec<u8>, String> {
    transform_request_degrading(value, LossyFeatureAllowlist::default()).map(|output| output.body)
}

/// Transform a parsed Anthropic Messages request body into Chat
/// Completions-compatible format, degrading the features the operator
/// allowlisted rather than rejecting them.
///
/// Every top-level field falls into one of these buckets:
/// - mapped fields are translated to their Chat Completions equivalent;
/// - allowlisted lossy features are stripped after their wire markers are validated, and the removal is recorded in
///   [`TransformOutput::degraded`];
/// - [`UNREPRESENTABLE_FIELDS`] reject the request with an error message, because the proxy would otherwise fabricate
///   their effect in the translated response;
/// - [`UNMAPPABLE_FIELDS`] reject when they carry a value and are not allowlisted;
/// - everything else is forwarded untouched, and the backend validates it.
///
/// A translated field always wins over a forwarded client key of the same
/// name. Returns the transformed JSON bytes and the degraded feature set, or
/// an error message.
#[expect(clippy::too_many_lines, reason = "linear request field extraction and mapping")]
pub(crate) fn transform_request_degrading(
    value: Value,
    allow: LossyFeatureAllowlist,
) -> Result<TransformOutput, String> {
    let Value::Object(mut body) = value else {
        return Err("request body is not a JSON object".to_owned());
    };

    // Degrade allowlisted features first: validate each feature's wire markers,
    // strip them in place, and record what was actually removed. The strict
    // validation below then runs on the cleaned body, so a leftover marker in an
    // unrecognized position still fails closed. The strippers walk only
    // Anthropic-structured positions and never descend into user-controlled tool
    // `input` or `input_schema` JSON (see [`degrade_prompt_caching`]).
    let mut degraded = DegradedFeatures::default();
    if allow.extended_thinking {
        degraded.extended_thinking = degrade_extended_thinking(&mut body)?;
    }
    if allow.prompt_caching {
        degraded.prompt_caching = degrade_prompt_caching(&mut body)?;
    }

    validate_faithful_request(&body)?;
    reject_unrepresentable_fields(&mut body)?;
    remove_null_unmappable_fields(&mut body);

    // Take every mapped field up front, in one place. Each becomes an owned
    // local that is moved into the helper emitting it.
    let model = body.remove("model");
    let max_tokens = body.remove("max_tokens");
    let system = body.remove("system");
    let messages = body.remove("messages");
    let stream = body.remove("stream");
    let stream_options = body.remove("stream_options");
    let stop_sequences = body.remove("stop_sequences");
    let temperature = body.remove("temperature");
    let top_p = body.remove("top_p");
    let tools = body.remove("tools");
    let tool_choice = body.remove("tool_choice");
    let had_tools = tools.is_some();

    let mut chat = Map::new();
    insert_if_some(&mut chat, "model", model);
    chat.insert("messages".to_owned(), build_messages(system, messages));
    insert_if_some(&mut chat, "max_completion_tokens", max_tokens);
    convert_stream(&mut chat, stream, stream_options);
    map_parameters(&mut chat, stop_sequences, temperature, top_p);
    map_metadata(&mut chat, body.remove("metadata"));
    map_output_config(&mut chat, body.remove("output_config"), body.remove("output_format"))?;
    convert_tools(&mut chat, tools);
    convert_parallel_tool_calls(&mut chat, tool_choice.as_ref());
    convert_tool_choice(&mut chat, tool_choice, had_tools);
    forward_unmapped_fields(&mut chat, body);

    let body = serde_json::to_vec(&Value::Object(chat)).map_err(|e| format!("serialization failed: {e}"))?;
    Ok(TransformOutput { body, degraded })
}

// -----------------------------------------------------------------------------
// Feature Degradation
// -----------------------------------------------------------------------------

/// Remove recognized extended-thinking markers, returning whether any were.
///
/// Validates the `thinking` request and any `context_management` before
/// removing them. A malformed `thinking` shape, or a `context_management` that
/// carries anything other than thinking-only edits, is rejected with a specific
/// 400 so unrepresentable intent never reaches the backend silently.
fn degrade_extended_thinking(body: &mut Map<String, Value>) -> Result<bool, String> {
    let mut degraded = false;

    let thinking_recognized = match body.get("thinking") {
        Some(value) if !value.is_null() => Some(thinking_request_is_recognized(value)),
        _ => None,
    };
    if let Some(recognized) = thinking_recognized {
        if !recognized {
            return Err("unsupported `thinking` shape for Chat Completions translation".to_owned());
        }
        thinking_budget_within_max_tokens(body)?;
        body.remove("thinking");
        degraded = true;
    }

    let edit_count = match body.get("context_management") {
        Some(value) if !value.is_null() => Some(recognized_thinking_edits(value)?),
        _ => None,
    };
    if let Some(count) = edit_count {
        body.remove("context_management");
        // An empty edits list removes nothing observable, so only a non-empty
        // thinking edit list counts as a real degradation.
        degraded |= count > 0;
    }

    Ok(degraded)
}

/// Whether a `thinking` request is a shape the translator recognizes.
///
/// Recognized, mirroring the Anthropic `ThinkingConfigParam` discriminator:
/// - `{"type": "enabled", "budget_tokens": <int ≥1024>, "display"?: <mode>}`
/// - `{"type": "adaptive", "display"?: <mode>}`
/// - `{"type": "disabled"}`
///
/// `budget_tokens` is required for `enabled` and must be an integer ≥1024 (the
/// schema minimum), and `display` must be a recognized mode (see
/// [`thinking_display_is_recognized`]). Any other shape is malformed to this
/// translator and fails closed.
fn thinking_request_is_recognized(value: &Value) -> bool {
    let Some(obj) = value.as_object() else { return false };
    match obj.get("type").and_then(Value::as_str) {
        Some("enabled") => {
            obj.keys()
                .all(|key| matches!(key.as_str(), "type" | "budget_tokens" | "display"))
                && obj
                    .get("budget_tokens")
                    .and_then(Value::as_u64)
                    .is_some_and(|budget| budget >= 1024)
                && thinking_display_is_recognized(obj.get("display"))
        },
        Some("adaptive") => {
            obj.keys().all(|key| matches!(key.as_str(), "type" | "display"))
                && thinking_display_is_recognized(obj.get("display"))
        },
        Some("disabled") => obj.keys().all(|key| key == "type"),
        _ => false,
    }
}

/// Whether an optional thinking `display` is a recognized `ThinkingDisplayMode`.
///
/// Absent or null selects the schema default (`summarized`); a present value
/// must be one of the defined modes. The marker is stripped, so this only keeps
/// a malformed value from being dropped silently.
fn thinking_display_is_recognized(display: Option<&Value>) -> bool {
    match display {
        None | Some(Value::Null) => true,
        Some(Value::String(mode)) => matches!(mode.as_str(), "summarized" | "omitted" | "updates"),
        Some(_) => false,
    }
}

/// Enforce the schema's cross-field rule that an `enabled` thinking
/// `budget_tokens` stays below the request `max_tokens`.
///
/// Only the `enabled` variant carries a budget; `adaptive`/`disabled` have none
/// to compare. When `max_tokens` is absent or non-numeric the backend owns that
/// rejection, so this checks only the pair the client actually sent. A
/// `budget_tokens` at or above `max_tokens` is malformed to the Anthropic schema
/// and fails closed rather than being stripped and reported as a clean
/// degradation.
fn thinking_budget_within_max_tokens(body: &Map<String, Value>) -> Result<(), String> {
    let Some(budget) = body
        .get("thinking")
        .and_then(Value::as_object)
        .filter(|thinking| thinking.get("type").and_then(Value::as_str) == Some("enabled"))
        .and_then(|thinking| thinking.get("budget_tokens"))
        .and_then(Value::as_u64)
    else {
        return Ok(());
    };
    let Some(max_tokens) = body.get("max_tokens").and_then(Value::as_u64) else {
        return Ok(());
    };
    if budget >= max_tokens {
        return Err(
            "`thinking.budget_tokens` must be less than `max_tokens` for Chat Completions translation".to_owned(),
        );
    }
    Ok(())
}

/// Count the edits in a `context_management` that carries only thinking edits.
///
/// Returns the edit count, or an error when the value is not an object, carries a
/// key other than `edits`, or whose `edits` is not an array of well-formed
/// `clear_thinking_20251015` edits (the only thinking-clearing edit the Anthropic
/// schema defines; see [`thinking_edit_is_recognized`]). `edits` is optional in
/// the schema (no `required`, `minItems: 0`), so an absent list — the no-op `{}` —
/// is a valid `context_management` with zero edits. Any other edit type (for
/// example `clear_tool_uses_*`), unknown edit field, or malformed `keep` is a
/// shape the translator cannot honor and is rejected rather than silently dropped.
fn recognized_thinking_edits(value: &Value) -> Result<usize, String> {
    let error = || "unsupported `context_management` for Chat Completions translation".to_owned();
    let obj = value.as_object().ok_or_else(error)?;
    if obj.keys().any(|key| key != "edits") {
        return Err(error());
    }
    // An absent `edits` is a valid zero-edit no-op. A present but non-array
    // `edits` (including explicit `null`, which the schema does not allow) is
    // malformed and fails closed.
    let Some(edits_value) = obj.get("edits") else {
        return Ok(0);
    };
    let edits = edits_value.as_array().ok_or_else(error)?;
    for edit in edits {
        if !thinking_edit_is_recognized(edit) {
            return Err(error());
        }
    }
    Ok(edits.len())
}

/// Whether a `context_management` edit is a well-formed `clear_thinking_20251015`.
///
/// Mirrors the Anthropic `ClearThinking20251015` schema: the object carries only
/// `type` (`clear_thinking_20251015`) and an optional `keep`, and `keep` — when
/// present — is a recognized shape (see [`clear_thinking_keep_is_recognized`]).
/// Any other edit type, unknown field, or malformed `keep` fails closed, because
/// the edit is stripped and a silently accepted malformed shape would be reported
/// as a clean degradation.
fn thinking_edit_is_recognized(edit: &Value) -> bool {
    let Some(obj) = edit.as_object() else { return false };
    obj.get("type").and_then(Value::as_str) == Some("clear_thinking_20251015")
        && obj.keys().all(|key| matches!(key.as_str(), "type" | "keep"))
        && clear_thinking_keep_is_recognized(obj.get("keep"))
}

/// Whether an optional `clear_thinking_20251015` `keep` is a recognized shape.
///
/// Mirrors the Anthropic schema's `keep` union: absent (the default), the string
/// `"all"`, or a discriminated `{type}` object selecting `thinking_turns` (with an
/// integer `value` ≥1) or `all`. The union carries no null, so an explicit
/// `"keep": null` — along with unknown fields or any other value — is unrecognized
/// and fails closed rather than being stripped as a clean degradation.
fn clear_thinking_keep_is_recognized(keep: Option<&Value>) -> bool {
    match keep {
        None => true,
        Some(Value::String(mode)) => mode == "all",
        Some(Value::Object(obj)) => match obj.get("type").and_then(Value::as_str) {
            Some("thinking_turns") => {
                obj.keys().all(|key| matches!(key.as_str(), "type" | "value"))
                    && obj.get("value").and_then(Value::as_u64).is_some_and(|value| value >= 1)
            },
            Some("all") => obj.keys().all(|key| key == "type"),
            _ => false,
        },
        Some(_) => false,
    }
}

/// Remove valid `cache_control` markers from every Anthropic-structured
/// position, returning whether any were removed.
///
/// Walks only the positions the Anthropic schema places cache markers in: the
/// top-level request `cache_control`, `system` text blocks, `tools` definitions,
/// message content blocks, and the nested text parts of a `tool_result`. It
/// never descends into a `tool_use` block's `input` or a tool's `input_schema`,
/// because a `cache_control` key there is user data, not Anthropic metadata. A
/// present but malformed marker is rejected with a specific 400.
fn degrade_prompt_caching(body: &mut Map<String, Value>) -> Result<bool, String> {
    let mut degraded = false;

    // The top-level `cache_control` is request-wide Anthropic cache metadata
    // with no Chat Completions equivalent, so it needs its own strip here rather
    // than riding a content block like the per-position markers below.
    strip_map_cache_control(body, &mut degraded)?;

    if let Some(Value::Array(blocks)) = body.get_mut("system") {
        for block in blocks.iter_mut() {
            strip_cache_control(block, &mut degraded)?;
        }
    }

    if let Some(Value::Array(tools)) = body.get_mut("tools") {
        for tool in tools.iter_mut() {
            strip_cache_control(tool, &mut degraded)?;
        }
    }

    if let Some(Value::Array(messages)) = body.get_mut("messages") {
        for message in messages.iter_mut() {
            let Some(Value::Array(blocks)) = message.get_mut("content") else {
                continue;
            };
            for block in blocks.iter_mut() {
                strip_cache_control(block, &mut degraded)?;
                // A tool_result's content is itself an array of Anthropic text
                // blocks, each a valid marker position.
                if block.get("type").and_then(Value::as_str) == Some("tool_result")
                    && let Some(Value::Array(parts)) = block.get_mut("content")
                {
                    for part in parts.iter_mut() {
                        strip_cache_control(part, &mut degraded)?;
                    }
                }
            }
        }
    }

    Ok(degraded)
}

/// Remove a recognized `cache_control` marker from one block in place.
///
/// Sets `degraded` when a valid marker is removed. A non-object block or an
/// absent/null marker is a no-op; a present but unrecognized marker is rejected.
fn strip_cache_control(block: &mut Value, degraded: &mut bool) -> Result<(), String> {
    let Some(obj) = block.as_object_mut() else {
        return Ok(());
    };
    strip_map_cache_control(obj, degraded)
}

/// Remove a recognized `cache_control` marker from an object map in place.
///
/// Shared by [`strip_cache_control`] (block positions) and the top-level request
/// marker. An absent or null marker is a no-op; a present but unrecognized marker
/// is rejected so a malformed value is never silently dropped.
fn strip_map_cache_control(obj: &mut Map<String, Value>, degraded: &mut bool) -> Result<(), String> {
    match obj.get("cache_control") {
        None | Some(Value::Null) => Ok(()),
        Some(marker) if cache_control_marker_is_recognized(marker) => {
            obj.remove("cache_control");
            *degraded = true;
            Ok(())
        },
        Some(_) => Err("unsupported `cache_control` marker for Chat Completions translation".to_owned()),
    }
}

/// Whether a `cache_control` value is a recognized Anthropic cache marker.
///
/// Recognized: `{"type": "ephemeral", "ttl"?: "5m" | "1h"}` (the only TTLs the
/// schema's `CacheControlEphemeral` enum defines). The marker is removed, so the
/// backend never validates it; this check keeps a malformed value from being
/// silently dropped.
fn cache_control_marker_is_recognized(marker: &Value) -> bool {
    let Some(obj) = marker.as_object() else {
        return false;
    };
    obj.get("type").and_then(Value::as_str) == Some("ephemeral")
        && obj.keys().all(|key| matches!(key.as_str(), "type" | "ttl"))
        && obj
            .get("ttl")
            .is_none_or(|ttl| matches!(ttl.as_str(), Some("5m" | "1h")))
}

/// Reject request data the translator would discard or change meaning.
#[expect(
    clippy::too_many_lines,
    reason = "sequential validation of the translated request shape"
)]
fn validate_faithful_request(body: &Map<String, Value>) -> Result<(), String> {
    for field in UNMAPPABLE_FIELDS {
        if body.get(field).is_some_and(|value| !value.is_null()) {
            return Err(format!("`{field}` cannot be translated to Chat Completions"));
        }
    }
    if let Some(system) = body.get("system") {
        match system {
            Value::String(_) | Value::Null => {},
            Value::Array(blocks) if blocks.iter().all(text_block_is_supported) => {},
            _ => return Err("`system` must be a string or array of text blocks".to_owned()),
        }
    }
    if body.get("stream").and_then(Value::as_bool) != Some(true)
        && body.get("stream_options").is_some_and(|value| !value.is_null())
    {
        return Err("`stream_options` requires `stream: true`".to_owned());
    }
    if let Some(options) = body.get("stream_options").filter(|value| !value.is_null()) {
        let options = options.as_object().ok_or("`stream_options` must be an object")?;
        if options
            .get("include_usage")
            .is_some_and(|value| !value.is_boolean() && !value.is_null())
        {
            return Err("`stream_options.include_usage` must be a boolean".to_owned());
        }
    }
    if let Some(metadata) = body.get("metadata").filter(|value| !value.is_null()) {
        let metadata = metadata.as_object().ok_or("`metadata` must be an object")?;
        if metadata.keys().any(|key| key != "user_id")
            || metadata
                .get("user_id")
                .is_some_and(|value| !value.is_null() && !value.is_string())
        {
            return Err("unsupported Anthropic `metadata` for Chat Completions translation".to_owned());
        }
    }
    if let Some(messages) = body.get("messages") {
        let messages = messages.as_array().ok_or("`messages` must be an array")?;
        for message in messages {
            if !has_only_keys(message, &["role", "content"]) {
                return Err("unsupported Anthropic message field for Chat Completions translation".to_owned());
            }
            let role = message
                .get("role")
                .and_then(Value::as_str)
                .ok_or("message is missing a string `role`")?;
            // Claude Code can put a text-only system message in this array.
            // Chat Completions represents it directly; the backend owns ordering.
            if !matches!(role, "system" | "user" | "assistant") {
                return Err(format!("unsupported Anthropic message role `{role}`"));
            }
            match message.get("content") {
                Some(Value::String(_)) => {},
                Some(Value::Array(blocks)) => validate_content_blocks(blocks, role)?,
                _ => return Err("message `content` must be a string or array".to_owned()),
            }
        }
    }
    if let Some(tools) = body.get("tools").filter(|tools| !tools.is_null()) {
        let tools = tools.as_array().ok_or("`tools` must be an array")?;
        for tool in tools {
            let tool = tool.as_object().ok_or("tool must be an object")?;
            let supported_type = match tool.get("type") {
                None => true,
                Some(Value::String(kind)) => kind == "custom",
                _ => false,
            };
            if !supported_type {
                return Err("typed Anthropic tools cannot be translated to Chat Completions".to_owned());
            }
            if !tool.get("name").is_some_and(Value::is_string) {
                return Err("tool requires a string `name`".to_owned());
            }
            if !tool.get("input_schema").is_some_and(Value::is_object) {
                return Err("tool requires an `input_schema` object".to_owned());
            }
            if tool.keys().any(|key| {
                !matches!(
                    key.as_str(),
                    "type" | "name" | "description" | "input_schema" | "strict"
                )
            }) || tool
                .get("description")
                .is_some_and(|value| !value.is_string() && !value.is_null())
                || tool
                    .get("strict")
                    .is_some_and(|value| !value.is_boolean() && !value.is_null())
            {
                return Err("unsupported Anthropic tool field for Chat Completions translation".to_owned());
            }
        }
    }
    if body
        .get("output_config")
        .is_some_and(|config| !config.is_null() && !config.is_object())
    {
        return Err("`output_config` must be an object".to_owned());
    }
    if let Some(choice) = body.get("tool_choice").filter(|choice| !choice.is_null()) {
        let valid = match choice {
            Value::String(kind) => matches!(kind.as_str(), "auto" | "any" | "none"),
            Value::Object(choice) => match choice.get("type").and_then(Value::as_str) {
                Some("auto" | "any" | "none") => choice
                    .keys()
                    .all(|key| matches!(key.as_str(), "type" | "disable_parallel_tool_use")),
                Some("tool") => {
                    choice.get("name").is_some_and(Value::is_string)
                        && choice
                            .keys()
                            .all(|key| matches!(key.as_str(), "type" | "name" | "disable_parallel_tool_use"))
                },
                _ => false,
            },
            _ => false,
        };
        let valid_parallel = choice
            .get("disable_parallel_tool_use")
            .is_none_or(|value| value.is_boolean() || value.is_null());
        let needs_tools = matches!(
            choice.get("type").and_then(Value::as_str).or_else(|| choice.as_str()),
            Some("any" | "tool")
        );
        let has_tools = body
            .get("tools")
            .and_then(Value::as_array)
            .is_some_and(|tools| !tools.is_empty());
        if !valid || !valid_parallel || (needs_tools && !has_tools) {
            return Err("unsupported Anthropic `tool_choice` for Chat Completions translation".to_owned());
        }
    }
    Ok(())
}

/// Validate content before lowering so the backend cannot accept a truncated request.
#[expect(clippy::too_many_lines, reason = "one validation arm per supported content block")]
fn validate_content_blocks(blocks: &[Value], role: &str) -> Result<(), String> {
    let has_tool_use = blocks
        .iter()
        .any(|block| block.get("type").and_then(Value::as_str) == Some("tool_use"));
    for block in blocks {
        match block.get("type").and_then(Value::as_str) {
            Some("text") if text_block_is_supported(block) => {},
            Some("image") if role == "user" && !has_tool_use => {
                let source = block.get("source").ok_or("image block is missing `source`")?;
                if !image_source_is_supported(source)
                    || !has_only_keys(block, &["type", "source", "cache_control"])
                    || !null_or_absent(block, "cache_control")
                {
                    return Err("image block has an unsupported `source`".to_owned());
                }
            },
            Some("tool_use") if role == "assistant" => {
                // The converter serializes any present JSON input as arguments;
                // an absent input retains its established empty-object default.
                if !block.get("id").is_some_and(Value::is_string)
                    || !block.get("name").is_some_and(Value::is_string)
                    || !has_only_keys(
                        block,
                        &["type", "id", "name", "input", "caller", "toolset_name", "cache_control"],
                    )
                    || !block.get("caller").is_none_or(direct_caller_is_supported)
                    || !null_or_absent(block, "toolset_name")
                    || !null_or_absent(block, "cache_control")
                {
                    return Err("tool_use block requires string `id` and `name`".to_owned());
                }
            },
            Some("tool_result") if role == "user" => {
                let text_content = match block.get("content") {
                    None | Some(Value::String(_)) => true,
                    Some(Value::Array(parts)) => parts.iter().all(text_block_is_supported),
                    _ => false,
                };
                if !block.get("tool_use_id").is_some_and(Value::is_string)
                    || !text_content
                    || block.get("is_error") == Some(&Value::Bool(true))
                    || block
                        .get("is_error")
                        .is_some_and(|value| !value.is_boolean() && !value.is_null())
                    || !has_only_keys(
                        block,
                        &[
                            "type",
                            "tool_use_id",
                            "content",
                            "is_error",
                            "toolset_name",
                            "cache_control",
                        ],
                    )
                    || !null_or_absent(block, "toolset_name")
                    || !null_or_absent(block, "cache_control")
                {
                    return Err("tool_result requires string `tool_use_id` and text `content`".to_owned());
                }
            },
            Some(kind) => return Err(format!("unsupported Anthropic content block `{kind}`")),
            None => return Err("Anthropic content block requires a string `type`".to_owned()),
        }
    }
    Ok(())
}

/// Check that a JSON object contains only fields the translator understands.
fn has_only_keys(value: &Value, keys: &[&str]) -> bool {
    value
        .as_object()
        .is_some_and(|fields| fields.keys().all(|key| keys.contains(&key.as_str())))
}

/// Null or empty citations carry no source information to preserve.
fn text_block_is_supported(block: &Value) -> bool {
    block.get("type").and_then(Value::as_str) == Some("text")
        && block.get("text").is_some_and(Value::is_string)
        && block
            .get("citations")
            .is_none_or(|citations| citations.is_null() || citations.as_array().is_some_and(Vec::is_empty))
        && null_or_absent(block, "cache_control")
        && has_only_keys(block, &["type", "text", "citations", "cache_control"])
}

/// Nullable request metadata has no effect when absent or null.
fn null_or_absent(value: &Value, key: &str) -> bool {
    value.get(key).is_none_or(Value::is_null)
}

/// A direct caller is the only caller form this translator emits and consumes.
fn direct_caller_is_supported(caller: &Value) -> bool {
    caller.is_null()
        || caller
            .as_object()
            .is_some_and(|fields| fields.len() == 1 && fields.get("type").and_then(Value::as_str) == Some("direct"))
}

/// Check the fields used by `convert_image_source` without copying image data.
fn image_source_is_supported(source: &Value) -> bool {
    match source.get("type").and_then(Value::as_str) {
        Some("base64") => {
            source.get("media_type").is_some_and(Value::is_string)
                && source.get("data").is_some_and(Value::is_string)
                && has_only_keys(source, &["type", "media_type", "data"])
        },
        Some("url") => source.get("url").is_some_and(Value::is_string) && has_only_keys(source, &["type", "url"]),
        _ => false,
    }
}

/// Forward every field the translation did not consume, leaving its
/// validation to the backend. A translated key always wins over a colliding
/// client key.
fn forward_unmapped_fields(chat: &mut Map<String, Value>, body: Map<String, Value>) {
    for (key, value) in body {
        chat.entry(key).or_insert(value);
    }
}

/// Remove null-only unmappable fields after validation treats them as absent.
fn remove_null_unmappable_fields(body: &mut Map<String, Value>) {
    for field in UNMAPPABLE_FIELDS {
        body.remove(field);
    }
}

/// Remove every [`UNREPRESENTABLE_FIELDS`] entry, failing on the first one
/// that carries a value. A JSON `null` or the field's documented default is
/// treated as absent.
fn reject_unrepresentable_fields(body: &mut Map<String, Value>) -> Result<(), String> {
    for field in UNREPRESENTABLE_FIELDS {
        if body
            .remove(field)
            .is_some_and(|value| !value.is_null() && !is_default_value(field, &value))
        {
            return Err(format!(
                "`{field}` is not supported when translating Anthropic Messages to Chat Completions"
            ));
        }
    }
    Ok(())
}

/// Whether `value` is the documented default of `field`, so that sending it
/// is indistinguishable from omitting it and the response misreports nothing.
///
/// The Chat Completions spec documents `n: 1`, `logprobs: false`,
/// `modalities: ["text"]` and `function_call: "none"` (the default when no
/// `functions` are present, which always holds since `functions` is
/// rejected). Anthropic documents `service_tier: "auto"`, and an empty
/// `mcp_servers` list configures nothing. The remaining rejected fields have
/// no default other than null.
fn is_default_value(field: &str, value: &Value) -> bool {
    match field {
        "n" => *value == 1,
        "logprobs" => *value == false,
        "modalities" => *value == json!(["text"]),
        "function_call" => *value == "none",
        "service_tier" => *value == "auto",
        "mcp_servers" => *value == json!([]),
        _ => false,
    }
}

// -----------------------------------------------------------------------------
// Field Consumption
// -----------------------------------------------------------------------------

/// Remove `key` from `map` and return the owned `String` when the value was a JSON string.
///
/// The entry is removed either way: a non-string value is dropped and `None` returned, matching the
/// `get(..).and_then(Value::as_str)` behavior, without re-allocating the string.
fn take_string(map: &mut Map<String, Value>, key: &str) -> Option<String> {
    match map.remove(key) {
        Some(Value::String(text)) => Some(text),
        _ => None,
    }
}

/// Move `value` into `target` under `key`, doing nothing when the field is absent.
///
/// Pairs with [`Map::remove`] so a mapped field travels from the parsed body into the translated body
/// without an intermediate copy.
fn insert_if_some(target: &mut Map<String, Value>, key: &str, value: Option<Value>) {
    if let Some(value) = value {
        target.insert(key.to_owned(), value);
    }
}

/// Build the Chat Completions `messages` array from the Anthropic `system` and
/// `messages` fields.
fn build_messages(system: Option<Value>, messages: Option<Value>) -> Value {
    let mut converted = Vec::new();
    hoist_system(&mut converted, system);
    convert_messages(&mut converted, messages);
    Value::Array(converted)
}

/// Build a Chat Completions message carrying plain string content.
fn text_message(role: String, content: String) -> Value {
    let mut message = Map::new();
    message.insert("role".to_owned(), Value::String(role));
    message.insert("content".to_owned(), Value::String(content));
    Value::Object(message)
}

/// Build a Chat Completions `text` content part, moving `text` into it.
fn text_content_part(text: String) -> Value {
    let mut part = Map::new();
    part.insert("type".to_owned(), Value::String("text".to_owned()));
    part.insert("text".to_owned(), Value::String(text));
    Value::Object(part)
}

/// Build a Chat Completions `image_url` content part, moving `url` into it.
fn image_content_part(url: String) -> Value {
    let mut image_url = Map::new();
    image_url.insert("url".to_owned(), Value::String(url));

    let mut part = Map::new();
    part.insert("type".to_owned(), Value::String("image_url".to_owned()));
    part.insert("image_url".to_owned(), Value::Object(image_url));
    Value::Object(part)
}

// -----------------------------------------------------------------------------
// System Message Hoisting
// -----------------------------------------------------------------------------

/// Hoist Anthropic top-level `system` to a Chat Completions system message.
fn hoist_system(messages: &mut Vec<Value>, system: Option<Value>) {
    let content = match system {
        Some(Value::String(text)) => text,
        Some(Value::Array(blocks)) => {
            let mut parts = Vec::new();
            for block in blocks {
                if let Value::Object(mut block) = block
                    && let Some(text) = take_string(&mut block, "text")
                {
                    parts.push(text);
                }
            }
            parts.join("\n")
        },
        _ => return,
    };

    if !content.is_empty() {
        messages.push(text_message("system".to_owned(), content));
    }
}

// -----------------------------------------------------------------------------
// Message Conversion
// -----------------------------------------------------------------------------

/// Convert Anthropic messages array to Chat Completions messages.
fn convert_messages(messages: &mut Vec<Value>, source: Option<Value>) {
    let Some(Value::Array(anthropic_messages)) = source else {
        return;
    };

    for msg in anthropic_messages {
        let Value::Object(mut msg) = msg else {
            continue;
        };
        let Some(role) = take_string(&mut msg, "role") else {
            continue;
        };

        match msg.remove("content") {
            Some(Value::String(text)) => {
                messages.push(text_message(role, text));
            },
            Some(Value::Array(blocks)) => {
                convert_content_blocks(messages, &role, blocks);
            },
            _ => {
                messages.push(text_message(role, String::new()));
            },
        }
    }
}

/// Convert typed content blocks to Chat Completions-compatible format.
/// Consumes the blocks to move their payloads into the translated message.
fn convert_content_blocks(messages: &mut Vec<Value>, role: &str, blocks: Vec<Value>) {
    let mut content_parts = Vec::new();
    let mut tool_calls = Vec::new();

    for block in blocks {
        let mut block = match block {
            Value::Object(block) => block,
            _ => Map::new(),
        };
        let block_type = take_string(&mut block, "type").unwrap_or_default();
        convert_single_block(block, &block_type, messages, role, &mut content_parts, &mut tool_calls);
    }

    finalize_content_blocks(messages, role, &mut content_parts, tool_calls);
}

/// Process a single content block within a message.
#[expect(
    clippy::too_many_arguments,
    reason = "accumulator pattern requires passing all state"
)]
fn convert_single_block(
    block: Map<String, Value>,
    block_type: &str,
    messages: &mut Vec<Value>,
    role: &str,
    content_parts: &mut Vec<Value>,
    tool_calls: &mut Vec<Value>,
) {
    match block_type {
        "text" => convert_text_block(block, content_parts),
        "image" => convert_image_block(block, content_parts),
        "search_result" => convert_search_result_block(block, content_parts),
        "document" => convert_document_block(block, content_parts),
        "tool_use" => convert_tool_use_block(block, tool_calls),
        "tool_result" => {
            flush_content_parts(messages, content_parts, role);
            convert_tool_result_block(block, messages);
        },
        "thinking" | "redacted_thinking" => {
            warn!(block_type, "dropping unsupported Anthropic content block");
        },
        _ => {
            warn!(block_type, "dropping unknown Anthropic content block type");
        },
    }
}

/// Convert a text content block.
fn convert_text_block(mut block: Map<String, Value>, content_parts: &mut Vec<Value>) {
    if let Some(text) = take_string(&mut block, "text") {
        content_parts.push(text_content_part(text));
    }
}

/// Convert an image content block.
fn convert_image_block(mut block: Map<String, Value>, content_parts: &mut Vec<Value>) {
    if let Some(source) = block.remove("source")
        && let Some(url_val) = convert_image_source(source)
    {
        content_parts.push(image_content_part(url_val));
    }
}

/// Convert a `search_result` block to backend-visible text context.
fn convert_search_result_block(block: Map<String, Value>, content_parts: &mut Vec<Value>) {
    if let Some(text) = flatten_search_result(block) {
        content_parts.push(text_content_part(text));
    }
}

/// Convert a `document` block to backend-visible text context.
fn convert_document_block(block: Map<String, Value>, content_parts: &mut Vec<Value>) {
    if let Some(text) = flatten_document(block) {
        content_parts.push(text_content_part(text));
    }
}

/// Take the string out of a `text` content part, leaving other parts untouched.
fn take_text_part(part: &mut Value) -> Option<String> {
    let part = part.as_object_mut()?;
    if part.get("type").and_then(Value::as_str) != Some("text") {
        return None;
    }
    take_string(part, "text")
}

/// Concatenate the text of every text content part, moving each string out.
fn take_joined_text(content_parts: &mut [Value]) -> Option<String> {
    let mut joined: Option<String> = None;
    for part in content_parts {
        let Some(text) = take_text_part(part) else {
            continue;
        };
        match &mut joined {
            None => joined = Some(text),
            Some(acc) => acc.push_str(&text),
        }
    }
    joined
}

/// Convert a `tool_use` content block to a Chat Completions tool call.
fn convert_tool_use_block(mut block: Map<String, Value>, tool_calls: &mut Vec<Value>) {
    let id = take_string(&mut block, "id").unwrap_or_default();
    let name = take_string(&mut block, "name").unwrap_or_default();

    let args = match block.remove("input") {
        Some(input) => serde_json::to_string(&input).unwrap_or_default(),
        None => "{}".to_owned(),
    };

    let mut function = Map::new();
    function.insert("name".to_owned(), Value::String(name));
    function.insert("arguments".to_owned(), Value::String(args));

    let mut tool_call = Map::new();
    tool_call.insert("id".to_owned(), Value::String(id));
    tool_call.insert("type".to_owned(), Value::String("function".to_owned()));
    tool_call.insert("function".to_owned(), Value::Object(function));
    tool_calls.push(Value::Object(tool_call));
}

/// Convert a `tool_result` content block to a Chat Completions tool message.
fn convert_tool_result_block(mut block: Map<String, Value>, messages: &mut Vec<Value>) {
    let tool_call_id = take_string(&mut block, "tool_use_id").unwrap_or_default();
    let is_error = block.get("is_error").and_then(Value::as_bool) == Some(true);

    let (mut result_content, image_content) = split_tool_result_content(block.remove("content"));

    if is_error {
        result_content = mark_tool_result_error(result_content);
    }

    let mut tool_message = Map::new();
    tool_message.insert("role".to_owned(), Value::String("tool".to_owned()));
    tool_message.insert("tool_call_id".to_owned(), Value::String(tool_call_id));
    tool_message.insert("content".to_owned(), Value::String(result_content));
    messages.push(Value::Object(tool_message));

    if !image_content.is_empty() {
        let mut image_message = Map::new();
        image_message.insert("role".to_owned(), Value::String("user".to_owned()));
        image_message.insert("content".to_owned(), Value::Array(image_content));
        messages.push(Value::Object(image_message));
    }
}

/// Emit the final message for accumulated content and tool calls.
fn finalize_content_blocks(
    messages: &mut Vec<Value>,
    role: &str,
    content_parts: &mut Vec<Value>,
    tool_calls: Vec<Value>,
) {
    if role == "assistant" && !tool_calls.is_empty() {
        let mut msg = Map::new();
        msg.insert("role".to_owned(), Value::String("assistant".to_owned()));
        if let Some(text) = take_joined_text(content_parts) {
            msg.insert("content".to_owned(), Value::String(text));
        }
        msg.insert("tool_calls".to_owned(), Value::Array(tool_calls));
        messages.push(Value::Object(msg));
    } else {
        flush_content_parts(messages, content_parts, role);
    }
}

/// Flush accumulated content parts as a message.
fn flush_content_parts(messages: &mut Vec<Value>, content_parts: &mut Vec<Value>, role: &str) {
    if content_parts.is_empty() {
        return;
    }

    let lone_text = match content_parts.as_mut_slice() {
        [part] => take_text_part(part),
        _ => None,
    };

    if let Some(text) = lone_text {
        messages.push(text_message(role.to_owned(), text));
    } else {
        let mut msg = Map::new();
        msg.insert("role".to_owned(), Value::String(role.to_owned()));
        msg.insert("content".to_owned(), Value::Array(std::mem::take(content_parts)));
        messages.push(Value::Object(msg));
    }

    content_parts.clear();
}

// -----------------------------------------------------------------------------
// Image Source Conversion
// -----------------------------------------------------------------------------

/// Convert Anthropic image source to an `image_url` URL string consuming the
/// source.
fn convert_image_source(source: Value) -> Option<String> {
    let Value::Object(mut source) = source else {
        return None;
    };
    let source_type = take_string(&mut source, "type")?;

    match source_type.as_str() {
        "base64" => {
            let media_type = take_string(&mut source, "media_type")?;
            let data = take_string(&mut source, "data")?;
            Some(format!("data:{media_type};base64,{data}"))
        },
        "url" => take_string(&mut source, "url"),
        _ => None,
    }
}

// -----------------------------------------------------------------------------
// Tool Result Content Extraction
// -----------------------------------------------------------------------------

/// Split a `tool_result` block's `content` into its flattened text and the
/// image parts promoted to a follow-up user message.
fn split_tool_result_content(content: Option<Value>) -> (String, Vec<Value>) {
    match content {
        Some(Value::String(text)) => (text, Vec::new()),
        Some(Value::Array(parts)) => split_tool_result_parts(parts),
        _ => (String::new(), Vec::new()),
    }
}

/// Split the parts of an array-form `tool_result.content`.
fn split_tool_result_parts(parts: Vec<Value>) -> (String, Vec<Value>) {
    let mut text_parts = Vec::new();
    let mut image_parts = Vec::new();

    for part in parts {
        let Value::Object(mut part) = part else {
            continue;
        };
        match take_string(&mut part, "type").as_deref() {
            Some("text") => {
                if let Some(text) = take_string(&mut part, "text") {
                    text_parts.push(text);
                }
            },
            Some("search_result") => {
                if let Some(text) = flatten_search_result(part) {
                    text_parts.push(text);
                }
            },
            Some("document") => {
                if let Some(text) = flatten_document(part) {
                    text_parts.push(text);
                }
            },
            Some("image") => convert_image_block(part, &mut image_parts),
            _ => {},
        }
    }

    (text_parts.join("\n"), image_parts)
}

/// Preserve Anthropic's `tool_result.is_error` semantic in text-only tool messages.
fn mark_tool_result_error(mut content: String) -> String {
    if content.is_empty() {
        "Anthropic tool_result error".to_owned()
    } else {
        content.insert_str(0, "Anthropic tool_result error:\n");
        content
    }
}

/// Flatten an Anthropic `search_result` block to plain text.
fn flatten_search_result(mut block: Map<String, Value>) -> Option<String> {
    let title = take_string(&mut block, "title").filter(|title| !title.is_empty());
    let source = take_string(&mut block, "source").filter(|source| !source.is_empty());
    let content = extract_text_blocks(block.remove("content"));

    if title.is_none() && source.is_none() && content.is_empty() {
        return None;
    }

    let mut flattened = String::new();

    if let Some(title) = title {
        flattened.push_str("Search result: ");
        flattened.push_str(&quote_label_value(&title));
    } else {
        flattened.push_str("Search result");
    }

    if let Some(source) = source {
        flattened.push_str("\nSource: ");
        flattened.push_str(&quote_label_value(&source));
    }

    if !content.is_empty() {
        flattened.push_str("\nContent:");
        for text in content {
            flattened.push('\n');
            flattened.push_str(&text);
        }
    }

    Some(flattened)
}

/// Flatten an Anthropic `document` block to plain text.
fn flatten_document(mut block: Map<String, Value>) -> Option<String> {
    let title = take_string(&mut block, "title").filter(|title| !title.is_empty());
    let context = take_string(&mut block, "context").filter(|context| !context.is_empty());
    let source_text = flatten_document_source(block.remove("source"));

    if title.is_none() && context.is_none() && source_text.is_none() {
        return None;
    }

    let mut flattened = String::new();

    if let Some(title) = title {
        flattened.push_str("Document: ");
        flattened.push_str(&quote_label_value(&title));
    } else {
        flattened.push_str("Document");
    }

    if let Some(context) = context {
        flattened.push_str("\nContext: ");
        flattened.push_str(&quote_label_value(&context));
    }

    if let Some(source_text) = source_text {
        flattened.push('\n');
        flattened.push_str(&source_text);
    }

    Some(flattened)
}

/// Flatten a `document.source` value to extractable text or a stable reference.
fn flatten_document_source(source: Option<Value>) -> Option<String> {
    let Value::Object(mut source) = source? else {
        return None;
    };
    let source_type = take_string(&mut source, "type")?;

    match source_type.as_str() {
        "text" => take_string(&mut source, "data")
            .filter(|data| !data.is_empty())
            .map(|data| format!("Content:\n{data}")),
        "content" => {
            let lines = extract_text_blocks(source.remove("content"));
            non_empty_lines(&lines).map(|content| format!("Content:\n{content}"))
        },
        "url" => take_string(&mut source, "url")
            .filter(|url| !url.is_empty())
            .map(|url| format!("Source: {}", quote_label_value(&url))),
        "file" => take_string(&mut source, "file_id")
            .filter(|file_id| !file_id.is_empty())
            .map(|file_id| format!("Source: {}", quote_label_value(&format!("file:{file_id}")))),
        "base64" => take_string(&mut source, "media_type")
            .filter(|media_type| !media_type.is_empty())
            .map(|media_type| format!("Source: {}", quote_label_value(&format!("base64:{media_type}")))),
        _ => None,
    }
}

/// Quote metadata values so embedded newlines cannot forge flattening labels.
fn quote_label_value(value: &str) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| format!("{value:?}"))
}

/// Extract text from an array of Anthropic text blocks, moving each string out.
fn extract_text_blocks(value: Option<Value>) -> Vec<String> {
    let Some(Value::Array(blocks)) = value else {
        return Vec::new();
    };

    let mut texts = Vec::new();
    for block in blocks {
        let Value::Object(mut block) = block else {
            continue;
        };
        if block.get("type").and_then(Value::as_str) != Some("text") {
            continue;
        }
        if let Some(text) = take_string(&mut block, "text")
            && !text.is_empty()
        {
            texts.push(text);
        }
    }
    texts
}

/// Join lines if at least one line contains content.
fn non_empty_lines(lines: &[String]) -> Option<String> {
    lines.iter().any(|line| !line.is_empty()).then(|| lines.join("\n"))
}

// -----------------------------------------------------------------------------
// Parameter Mapping
// -----------------------------------------------------------------------------

/// Move `stream` through and request streaming usage when enabled.
fn convert_stream(chat: &mut Map<String, Value>, stream: Option<Value>, stream_options: Option<Value>) {
    let Some(stream) = stream else {
        return;
    };

    let streaming = stream.as_bool() == Some(true);
    chat.insert("stream".to_owned(), stream);

    if !streaming {
        return;
    }

    let mut opts = match stream_options {
        Some(Value::Object(opts)) => opts,
        _ => Map::new(),
    };
    opts.insert("include_usage".to_owned(), Value::Bool(true));
    chat.insert("stream_options".to_owned(), Value::Object(opts));
}

/// Map Anthropic parameters to Chat Completions-compatible equivalents.
fn map_parameters(
    chat: &mut Map<String, Value>,
    stop_sequences: Option<Value>,
    temperature: Option<Value>,
    top_p: Option<Value>,
) {
    insert_if_some(chat, "stop", stop_sequences);
    insert_if_some(chat, "temperature", temperature);
    insert_if_some(chat, "top_p", top_p);
}

/// Map Anthropic `metadata.user_id` to Chat Completions `safety_identifier`,
/// the field with the same abuse-detection purpose.
///
/// The identifier is sent as its SHA-256 hex digest: Anthropic allows up to
/// 512 characters while `safety_identifier` allows 64, and the digest is
/// exactly 64. A null `user_id` is omitted because `safety_identifier` is not
/// nullable. Chat Completions has its own `metadata` field with different
/// semantics, so the Anthropic object itself never travels.
fn map_metadata(chat: &mut Map<String, Value>, metadata: Option<Value>) {
    if let Some(Value::Object(mut metadata)) = metadata
        && let Some(user_id) = take_string(&mut metadata, "user_id")
    {
        let hex = hash::hex(&Sha256::digest(user_id.as_bytes()));
        chat.insert("safety_identifier".to_owned(), Value::String(hex));
    }
}

/// Map Anthropic `output_config` to the Chat Completions controls with the
/// same meaning: `effort` to `reasoning_effort`, whose enum contains every
/// Anthropic level, and a `json_schema` `format` to a strict
/// `response_format`, since Anthropic structured outputs guarantee schema
/// conformance. Any other `output_config` key (such as the beta
/// `task_budget`) has no Chat Completions equivalent and rejects the request
/// rather than being silently discarded.
///
/// The deprecated top-level `output_format` is the older spelling of
/// `output_config.format`.
fn map_output_config(
    chat: &mut Map<String, Value>,
    output_config: Option<Value>,
    output_format: Option<Value>,
) -> Result<(), String> {
    let mut config = match output_config {
        Some(Value::Object(config)) => config,
        _ => Map::new(),
    };
    // Each known key is nullable in the schema, and null means unset. Nulls
    // under unknown keys stay, so they reach the unsupported-key rejection
    // below, as the schema's `additionalProperties: false` would reject them.
    for key in OUTPUT_CONFIG_KEYS {
        if config.get(key).is_some_and(Value::is_null) {
            config.remove(key);
        }
    }
    insert_if_some(chat, "reasoning_effort", config.remove("effort"));
    if let Some(format) = config
        .remove("format")
        .or(output_format)
        .filter(|format| !format.is_null())
    {
        chat.insert("response_format".to_owned(), response_format(format)?);
    }
    match config.keys().next() {
        Some(key) => Err(format!(
            "`output_config.{key}` is not supported when translating Anthropic Messages to Chat Completions"
        )),
        None => Ok(()),
    }
}

/// Build a strict Chat Completions `json_schema` response format from an
/// Anthropic `json_schema` output format.
fn response_format(format: Value) -> Result<Value, String> {
    if let Value::Object(mut format) = format
        && format.get("type").and_then(Value::as_str) == Some("json_schema")
        && let Some(schema) = format.remove("schema")
    {
        return Ok(json!({
            "type": "json_schema",
            "json_schema": {"name": "output_format", "strict": true, "schema": schema},
        }));
    }
    Err("`output_config.format` must be a `json_schema` object with a `schema` when translating Anthropic Messages to Chat Completions".to_owned())
}

// -----------------------------------------------------------------------------
// Tool Conversion
// -----------------------------------------------------------------------------

/// Convert Anthropic tool definitions to Chat Completions function tools.
fn convert_tools(chat: &mut Map<String, Value>, tools: Option<Value>) {
    let Some(Value::Array(tools)) = tools else {
        return;
    };

    let mut chat_tools = Vec::new();

    for tool in tools {
        if let Some(chat_tool) = convert_tool_definition(tool) {
            chat_tools.push(chat_tool);
        }
    }

    if !chat_tools.is_empty() {
        chat.insert("tools".to_owned(), Value::Array(chat_tools));
    }
}

/// Return a stable JSON type name for diagnostics, never the value itself.
fn json_type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

/// Classify an Anthropic tool: keep only untyped or explicit `custom` client
/// tools, dropping (and logging) every typed server tool so unknown server
/// tools fail closed instead of leaking to the backend as client functions.
fn is_translatable_client_tool(tool: &Value) -> bool {
    match tool.get("type") {
        None => true,
        Some(Value::String(tool_type)) if tool_type == "custom" => true,
        Some(Value::String(tool_type)) => {
            warn!(tool_type, "dropping typed Anthropic tool");
            false
        },
        Some(other) => {
            // Log only the JSON value kind, never the value itself: `type` is
            // attacker-controlled and could carry a large or sensitive payload.
            warn!(
                type_kind = json_type_name(other),
                "dropping Anthropic tool with non-string type"
            );
            false
        },
    }
}

/// Convert one Anthropic client tool definition to a Chat Completions tool.
/// Consumes the definition so `input_schema` moves into the generated function parameters.
fn convert_tool_definition(tool: Value) -> Option<Value> {
    if !is_translatable_client_tool(&tool) {
        return None;
    }

    let mut tool = match tool {
        Value::Object(fields) => fields,
        _ => Map::new(),
    };

    let name = take_string(&mut tool, "name").unwrap_or_default();
    let description = take_string(&mut tool, "description").unwrap_or_default();
    let parameters = tool.remove("input_schema").unwrap_or_else(|| json!({"type": "object"}));
    let strict = tool.get("strict").and_then(Value::as_bool);

    let mut function = Map::new();
    function.insert("name".to_owned(), Value::String(name));
    function.insert("description".to_owned(), Value::String(description));
    function.insert("parameters".to_owned(), parameters);
    if let Some(strict) = strict {
        function.insert("strict".to_owned(), Value::Bool(strict));
    }

    let mut chat_tool = Map::new();
    chat_tool.insert("type".to_owned(), Value::String("function".to_owned()));
    chat_tool.insert("function".to_owned(), Value::Object(function));
    Some(Value::Object(chat_tool))
}

// -----------------------------------------------------------------------------
// Tool Choice Conversion
// -----------------------------------------------------------------------------

/// Convert Anthropic `disable_parallel_tool_use` to Chat Completions format.
fn convert_parallel_tool_calls(chat: &mut Map<String, Value>, tool_choice: Option<&Value>) {
    let Some(Value::Object(tool_choice)) = tool_choice else {
        return;
    };

    if tool_choice
        .get("disable_parallel_tool_use")
        .and_then(Value::as_bool)
        .is_some_and(|disabled| disabled)
    {
        chat.insert("parallel_tool_calls".to_owned(), Value::Bool(false));
    }
}

/// Convert Anthropic `tool_choice` to Chat Completions format.
fn convert_tool_choice(chat: &mut Map<String, Value>, tool_choice: Option<Value>, had_tools: bool) {
    let Some(tool_choice) = tool_choice else {
        return;
    };

    if had_tools && !chat.contains_key("tools") {
        return;
    }

    let chat_choice = match tool_choice {
        Value::String(keyword) => Value::String(tool_choice_keyword(&keyword).to_owned()),
        Value::Object(tool_choice) => object_tool_choice(tool_choice),
        _ => return,
    };

    chat.insert("tool_choice".to_owned(), chat_choice);
}

/// Map an Anthropic `tool_choice` type to its Chat Completions keyword.
fn tool_choice_keyword(anthropic: &str) -> &'static str {
    match anthropic {
        "any" => "required",
        "none" => "none",
        _ => "auto",
    }
}

/// Convert an object-form `tool_choice`, moving a named tool's name through.
fn object_tool_choice(mut tool_choice: Map<String, Value>) -> Value {
    let names_a_tool = tool_choice.get("type").and_then(Value::as_str) == Some("tool");

    if names_a_tool && let Some(name) = take_string(&mut tool_choice, "name") {
        let mut function = Map::new();
        function.insert("name".to_owned(), Value::String(name));

        let mut choice = Map::new();
        choice.insert("type".to_owned(), Value::String("function".to_owned()));
        choice.insert("function".to_owned(), Value::Object(function));
        return Value::Object(choice);
    }

    let kind = tool_choice.get("type").and_then(Value::as_str).unwrap_or_default();
    Value::String(tool_choice_keyword(kind).to_owned())
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
#[expect(clippy::unwrap_used, clippy::indexing_slicing, reason = "tests")]
mod tests {
    use super::*;

    /// Parse a raw request body and transform it, mirroring the filter's
    /// parse-once call path including its parse-error message.
    fn transform_bytes(body: &[u8]) -> Result<Vec<u8>, String> {
        let value = serde_json::from_slice(body).map_err(|e| format!("invalid JSON: {e}"))?;
        transform_request(value)
    }

    #[test]
    #[expect(clippy::too_many_lines, reason = "enumerates semantic loss cases at one boundary")]
    fn strict_translation_rejects_semantic_losses() {
        for (request, field) in [
            (json!({"thinking": {"type": "enabled"}}), "thinking"),
            (json!({"context_management": {"edits": []}}), "context_management"),
            (json!({"system": 42}), "system"),
            (json!({"messages": [{"role": "user", "content": 42}]}), "content"),
            (
                json!({"messages": [{"role": "user", "content": [{"type": "unknown"}]}]}),
                "unknown",
            ),
            (
                json!({"messages": [{"role": "assistant", "content": [
                    {"type": "image", "source": {"type": "url", "url": "https://example.test/i.png"}},
                    {"type": "tool_use", "id": "c1", "name": "f", "input": {}}
                ]}]}),
                "image",
            ),
            (
                json!({"tools": [{"type": "web_search_20250305", "name": "web_search"}]}),
                "typed",
            ),
            (
                json!({"messages": [{"role": "user", "content": [{"type": "text", "text": "Hi", "cache_control": {"type": "ephemeral"}}]}]}),
                "text",
            ),
            (json!({"output_config": 42}), "output_config"),
            (
                json!({"stream": true, "stream_options": "include_usage"}),
                "stream_options",
            ),
        ] {
            let mut body = json!({"model": "m", "messages": [{"role": "user", "content": "Hi"}]});
            body.as_object_mut()
                .unwrap()
                .extend(request.as_object().unwrap().clone());
            let error = transform_bytes(body.to_string().as_bytes()).unwrap_err();
            assert!(error.contains(field), "{field}: {error}");
        }
    }

    #[test]
    fn strict_translation_accepts_generated_history_and_text_tool_results() {
        let body = json!({
            "model": "m",
            "messages": [
                {"role": "assistant", "content": [
                    {"type": "text", "text": "Hello", "citations": null, "cache_control": null},
                    {"type": "tool_use", "id": "call_1", "name": "lookup", "input": {}, "caller": {"type": "direct"}, "toolset_name": null}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "call_1", "toolset_name": null, "content": [
                        {"type": "text", "text": "Sunny", "citations": null}
                    ]}
                ]}
            ]
        });
        let translated: Value = serde_json::from_slice(&transform_request(body).unwrap()).unwrap();
        assert_eq!(translated["messages"][0]["content"], "Hello");
        assert_eq!(translated["messages"][0]["tool_calls"][0]["function"]["name"], "lookup");
        assert_eq!(translated["messages"][1]["content"], "Sunny");
    }

    #[test]
    fn strict_translation_accepts_empty_citations_in_history() {
        let body = json!({
            "model": "m",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Hi", "citations": []}]},
                {"role": "assistant", "content": [{"type": "text", "text": "Hello", "citations": []}]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": [
                    {"type": "text", "text": "Sunny", "citations": []}
                ]}]}
            ]
        });
        let translated: Value = serde_json::from_slice(&transform_request(body).unwrap()).unwrap();
        assert_eq!(translated["messages"][0]["content"], "Hi");
        assert_eq!(translated["messages"][1]["content"], "Hello");
        assert_eq!(translated["messages"][2]["content"], "Sunny");
    }

    #[test]
    fn strict_translation_preserves_system_message_in_history() {
        let body = json!({
            "model": "m",
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": "Be concise"}]},
                {"role": "user", "content": "Hi"}
            ]
        });
        let translated: Value = serde_json::from_slice(&transform_request(body).unwrap()).unwrap();
        assert_eq!(
            translated["messages"][0],
            json!({"role": "system", "content": "Be concise"})
        );
        assert_eq!(translated["messages"][1], json!({"role": "user", "content": "Hi"}));
    }

    #[test]
    fn strict_translation_treats_omitted_tool_result_content_as_empty() {
        let body = json!({
            "model": "m",
            "messages": [{"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "call_1"}
            ]}]
        });
        let translated: Value = serde_json::from_slice(&transform_request(body).unwrap()).unwrap();
        assert_eq!(translated["messages"][0]["role"], "tool");
        assert_eq!(translated["messages"][0]["content"], "");
    }

    #[test]
    fn strict_translation_rejects_nonempty_citations_and_indirect_caller() {
        for block in [
            json!({"type": "text", "text": "Hello", "citations": [{"url": "https://example.test"}]}),
            json!({"type": "tool_use", "id": "call_1", "name": "lookup", "input": {}, "caller": {"type": "server"}}),
        ] {
            let body = json!({"model": "m", "messages": [{"role": "assistant", "content": [block]}]});
            assert!(transform_request(body).is_err());
        }
    }

    #[test]
    fn basic_text_request() {
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"messages":[{"role":"user","content":"Hello"}]}"#;
        let result = transform_bytes(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        assert_eq!(parsed["model"], "claude-opus-4-8", "model preserved");
        assert_eq!(
            parsed["max_completion_tokens"], 1024,
            "max_tokens mapped to max_completion_tokens"
        );
        assert!(
            parsed.get("max_tokens").is_none(),
            "max_tokens must not appear in output"
        );
        assert_eq!(parsed["messages"][0]["role"], "user", "user message role");
        assert_eq!(parsed["messages"][0]["content"], "Hello", "user message content");
    }

    #[test]
    fn mapped_fields_keep_a_stable_serialized_key_order() {
        // `serde_json` runs with `preserve_order`, so the order fields are
        // emitted in `transform_request` is the order sent upstream. Pin it so
        // reordering the emission sequence cannot silently reshape the wire body.
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"stream":true,"stop_sequences":["x"],"temperature":0.5,"top_p":0.9,"top_k":40,"tools":[{"name":"t","input_schema":{"type":"object"}}],"tool_choice":{"type":"any","disable_parallel_tool_use":true},"messages":[{"role":"user","content":"Hi"}]}"#;
        let result = transform_bytes(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        let keys: Vec<&str> = parsed.as_object().unwrap().keys().map(String::as_str).collect();
        assert_eq!(
            keys,
            vec![
                "model",
                "messages",
                "max_completion_tokens",
                "stream",
                "stream_options",
                "stop",
                "temperature",
                "top_p",
                "tools",
                "parallel_tool_calls",
                "tool_choice",
                "top_k",
            ],
            "translated request key order must stay stable"
        );
    }

    #[test]
    fn tool_input_schema_is_preserved_verbatim() {
        let body = br#"{"model":"m","tools":[{"name":"t","description":"d","input_schema":{"type":"object","properties":{"a":{"type":"array","items":{"type":"string"}},"b":{"enum":[1,2,3]}},"required":["a"],"additionalProperties":false}}],"messages":[{"role":"user","content":"Hi"}]}"#;
        let result = transform_bytes(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        assert_eq!(
            parsed["tools"][0]["function"]["parameters"],
            json!({
                "type": "object",
                "properties": {"a": {"type": "array", "items": {"type": "string"}}, "b": {"enum": [1, 2, 3]}},
                "required": ["a"],
                "additionalProperties": false
            }),
            "moving the schema must not alter its contents"
        );
    }

    #[test]
    fn tool_definition_with_non_string_name_is_rejected() {
        let body = br#"{"model":"m","tools":[{"name":42,"description":true,"input_schema":{"type":"object"}}],"messages":[{"role":"user","content":"Hi"}]}"#;
        let error = transform_bytes(body).unwrap_err();
        assert!(error.contains("string `name`"), "{error}");
    }

    #[test]
    fn stream_options_without_streaming_are_rejected() {
        let body = br#"{"model":"m","stream":false,"stream_options":{"include_usage":false},"messages":[{"role":"user","content":"Hi"}]}"#;
        let error = transform_bytes(body).unwrap_err();
        assert!(error.contains("`stream_options` requires"), "{error}");
    }

    #[test]
    fn caller_stream_options_are_preserved_alongside_include_usage() {
        let body =
            br#"{"model":"m","stream":true,"stream_options":{"custom":1},"messages":[{"role":"user","content":"Hi"}]}"#;
        let result = transform_bytes(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        assert_eq!(
            parsed["stream_options"]["custom"], 1,
            "caller options are moved through"
        );
        assert_eq!(parsed["stream_options"]["include_usage"], true);
    }

    #[test]
    fn system_hoisted() {
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"system":"Be helpful.","messages":[{"role":"user","content":"Hi"}]}"#;
        let result = transform_bytes(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        assert_eq!(
            parsed["messages"][0]["role"], "system",
            "system message should be first"
        );
        assert_eq!(parsed["messages"][0]["content"], "Be helpful.", "system content");
        assert_eq!(parsed["messages"][1]["role"], "user", "user message follows system");
    }

    #[test]
    fn system_text_blocks_joined() {
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"system":[{"type":"text","text":"Part 1"},{"type":"text","text":"Part 2"}],"messages":[{"role":"user","content":"Hi"}]}"#;
        let result = transform_bytes(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        assert_eq!(
            parsed["messages"][0]["content"], "Part 1\nPart 2",
            "text blocks should be joined"
        );
    }

    #[test]
    fn tool_use_converted() {
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"messages":[{"role":"assistant","content":[{"type":"tool_use","id":"call_1","name":"get_weather","input":{"city":"NYC"}}]}]}"#;
        let result = transform_bytes(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        let msg = &parsed["messages"][0];
        assert_eq!(msg["role"], "assistant", "assistant role");
        assert_eq!(msg["tool_calls"][0]["function"]["name"], "get_weather", "tool name");
        assert!(
            msg["tool_calls"][0]["function"]["arguments"]
                .as_str()
                .unwrap()
                .contains("NYC"),
            "tool arguments contain city"
        );
    }

    #[test]
    fn tool_result_converted() {
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"call_1","content":"72F sunny"}]}]}"#;
        let result = transform_bytes(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        assert_eq!(parsed["messages"][0]["role"], "tool", "tool role");
        assert_eq!(parsed["messages"][0]["tool_call_id"], "call_1", "tool_call_id");
        assert_eq!(parsed["messages"][0]["content"], "72F sunny", "tool result content");
    }

    #[test]
    fn tool_result_error_is_rejected() {
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"call_1","content":"cat: missing.txt: No such file or directory","is_error":true}]}]}"#;
        let error = transform_bytes(body).unwrap_err();
        assert!(error.contains("tool_result"), "{error}");
    }

    #[test]
    fn tool_result_image_is_rejected() {
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"call_1","content":[{"type":"text","text":"chart"},{"type":"image","source":{"type":"url","url":"https://example.com/chart.png"}}]}]}]}"#;
        let error = transform_bytes(body).unwrap_err();
        assert!(error.contains("tool_result"), "{error}");
    }

    #[test]
    fn top_level_search_result_is_rejected() {
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"messages":[{"role":"user","content":[{"type":"search_result","source":"https://docs.example.test/product","title":"Product Guide","content":[{"type":"text","text":"The default timeout is 30 seconds."},{"type":"text","text":"The maximum timeout is 120 seconds."}]},{"type":"text","text":"What is the timeout range?"}]}]}"#;
        let error = transform_bytes(body).unwrap_err();
        assert!(error.contains("search_result"), "{error}");
    }

    #[test]
    fn tool_result_search_result_is_rejected() {
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"call_1","content":[{"type":"search_result","source":"kb://timeouts","title":"Timeout KB","content":[{"type":"text","text":"Timeouts default to 30 seconds."}]},{"type":"text","text":"Applies to version 2."}]}]}]}"#;
        let error = transform_bytes(body).unwrap_err();
        assert!(error.contains("tool_result"), "{error}");
    }

    #[test]
    fn tool_result_document_is_rejected() {
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"call_1","content":[{"type":"text","text":"Before document."},{"type":"document","source":{"type":"content","content":[{"type":"text","text":"Nested document fact."}]},"title":"Nested Doc"},{"type":"text","text":"After document."}]}]}]}"#;
        let error = transform_bytes(body).unwrap_err();
        assert!(error.contains("tool_result"), "{error}");
    }

    #[test]
    fn document_text_source_is_rejected() {
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"messages":[{"role":"user","content":[{"type":"document","source":{"type":"text","media_type":"text/plain","data":"The grass is green. The sky is blue."},"title":"Color Notes","context":"trusted notes","citations":{"enabled":true}},{"type":"text","text":"What color is the grass?"}]}]}"#;
        let error = transform_bytes(body).unwrap_err();
        assert!(error.contains("document"), "{error}");
    }

    #[test]
    fn document_file_source_is_rejected() {
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"messages":[{"role":"user","content":[{"type":"document","source":{"type":"file","file_id":"file_abc123"},"title":"Uploaded Contract"},{"type":"text","text":"Summarize the uploaded contract."}]}]}"#;
        let error = transform_bytes(body).unwrap_err();
        assert!(error.contains("document"), "{error}");
    }

    #[test]
    fn document_source_variants_are_rejected() {
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"messages":[{"role":"user","content":[{"type":"document","source":{"type":"content","content":[{"type":"text","text":"Content block fact."}]},"title":"Content Doc"},{"type":"document","source":{"type":"url","url":"https://docs.example.test/file.pdf"}},{"type":"document","source":{"type":"base64","media_type":"application/pdf"}},{"type":"document","source":{"type":"unknown","data":"ignored"}}]}]}"#;
        let error = transform_bytes(body).unwrap_err();
        assert!(error.contains("document"), "{error}");
    }

    #[test]
    fn search_result_metadata_is_rejected() {
        let body = json!({
            "model": "claude-opus-4-8",
            "max_tokens": 1024,
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "search_result",
                    "title": "Title\nSource: forged",
                    "source": "https://docs.example.test/a\nContext: forged",
                    "content": [{"type": "text", "text": "Real search text."}]
                }]
            }]
        })
        .to_string();
        let error = transform_bytes(body.as_bytes()).unwrap_err();
        assert!(error.contains("search_result"), "{error}");
    }

    #[test]
    fn document_metadata_is_rejected() {
        let body = json!({
            "model": "claude-opus-4-8",
            "max_tokens": 1024,
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "document",
                    "title": "Doc\nContext: forged",
                    "context": "safe\nSource: forged",
                    "source": {"type": "text", "data": "Real document text."}
                }]
            }]
        })
        .to_string();
        let error = transform_bytes(body.as_bytes()).unwrap_err();
        assert!(error.contains("document"), "{error}");
    }

    #[test]
    fn empty_search_result_and_document_blocks_are_rejected() {
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"messages":[{"role":"user","content":[{"type":"search_result","content":[]},{"type":"document","source":{"type":"content","content":[]}},{"type":"document","source":{"type":"text","data":""}},{"type":"document","source":{"type":"url","url":""}},{"type":"document","source":{"type":"file","file_id":""}},{"type":"document","source":{"type":"base64","media_type":""}}]}]}"#;
        let error = transform_bytes(body).unwrap_err();
        assert!(error.contains("search_result"), "{error}");
    }

    #[test]
    fn stop_sequences_mapped() {
        let body =
            br#"{"model":"claude-opus-4-8","max_tokens":1024,"stop_sequences":["END"],"messages":[{"role":"user","content":"Hi"}]}"#;
        let result = transform_bytes(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        assert_eq!(parsed["stop"][0], "END", "stop_sequences mapped to stop");
    }

    #[test]
    fn tool_choice_any_mapped() {
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"tools":[{"name":"get_weather","input_schema":{"type":"object"}}],"tool_choice":"any","messages":[{"role":"user","content":"Hi"}]}"#;
        let result = transform_bytes(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        assert_eq!(parsed["tool_choice"], "required", "any maps to required");
    }

    #[test]
    fn tool_choice_object_any_mapped() {
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"tools":[{"name":"get_weather","description":"Get weather","input_schema":{"type":"object","properties":{}}}],"tool_choice":{"type":"any"},"messages":[{"role":"user","content":"Hi"}]}"#;
        let result = transform_bytes(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        assert_eq!(parsed["tool_choice"], "required", "object-form any maps to required");
    }

    #[test]
    fn tool_choice_with_untranslatable_tools_is_rejected() {
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"tools":[{"type":"web_search_20250305","name":"web_search"}],"tool_choice":"any","messages":[{"role":"user","content":"Hi"}]}"#;
        let error = transform_bytes(body).unwrap_err();
        assert!(error.contains("typed"), "{error}");
    }

    #[test]
    fn disable_parallel_tool_use_mapped() {
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"tools":[{"name":"get_weather","description":"Get weather","input_schema":{"type":"object","properties":{}}}],"tool_choice":{"type":"auto","disable_parallel_tool_use":true},"messages":[{"role":"user","content":"Hi"}]}"#;
        let result = transform_bytes(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        assert_eq!(
            parsed["parallel_tool_calls"], false,
            "disable_parallel_tool_use should disable parallel tool calls"
        );
    }

    #[test]
    fn tool_definitions_converted() {
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"tools":[{"name":"get_weather","description":"Get weather","input_schema":{"type":"object","properties":{"city":{"type":"string"}}}}],"messages":[{"role":"user","content":"Hi"}]}"#;
        let result = transform_bytes(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        assert_eq!(parsed["tools"][0]["type"], "function", "tool type should be function");
        assert_eq!(parsed["tools"][0]["function"]["name"], "get_weather", "tool name");
    }

    #[test]
    fn tool_definition_strict_mapped() {
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"tools":[{"name":"get_weather","description":"Get weather","input_schema":{"type":"object","properties":{"city":{"type":"string"}}},"strict":true},{"name":"get_time","description":"Get time","input_schema":{"type":"object"},"strict":false}],"messages":[{"role":"user","content":"Hi"}]}"#;
        let result = transform_bytes(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        assert_eq!(
            parsed["tools"][0]["function"]["strict"], true,
            "Anthropic strict true should map to Chat Completions function strict"
        );
        assert_eq!(
            parsed["tools"][1]["function"]["strict"], false,
            "Anthropic strict false should remain false"
        );
    }

    #[test]
    fn image_base64_converted() {
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"messages":[{"role":"user","content":[{"type":"image","source":{"type":"base64","media_type":"image/jpeg","data":"abc123"}},{"type":"text","text":"What is this?"}]}]}"#;
        let result = transform_bytes(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        let content = &parsed["messages"][0]["content"];
        assert_eq!(content[0]["type"], "image_url", "image type");
        assert_eq!(
            content[0]["image_url"]["url"], "data:image/jpeg;base64,abc123",
            "data URL"
        );
        assert_eq!(content[1]["type"], "text", "text part follows");
    }

    #[test]
    fn unmapped_fields_are_forwarded_untouched() {
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"speed":"fast","messages":[{"role":"user","content":"Hi"}]}"#;
        let result = transform_bytes(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        assert_eq!(parsed["speed"], "fast", "unmapped field must reach the backend");
    }

    #[test]
    fn forwarded_fields_never_overwrite_translated_ones() {
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"max_completion_tokens":999,"messages":[{"role":"user","content":"Hi"}]}"#;
        let result = transform_bytes(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        assert_eq!(
            parsed["max_completion_tokens"], 1024,
            "translated value must win over a colliding client key"
        );
    }

    #[test]
    fn metadata_user_id_is_hashed_into_safety_identifier() {
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"metadata":{"user_id":"user-1"},"messages":[{"role":"user","content":"Hi"}]}"#;
        let result = transform_bytes(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        // SHA-256 of "user-1", as `sha256sum` prints it.
        let expected = "c6c289e49e9c05b2145860387b73bcb18df43fb09a1e4a4a9713c76c88bb541b";
        assert_eq!(expected.len(), 64, "digest fits the 64-character limit");
        assert_eq!(parsed["safety_identifier"], expected, "user_id hashed");
        assert!(
            parsed.get("metadata").is_none(),
            "Anthropic metadata must not reach a Chat Completions backend"
        );
    }

    #[test]
    fn metadata_null_user_id_is_omitted() {
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"metadata":{"user_id":null},"messages":[{"role":"user","content":"Hi"}]}"#;
        let result = transform_bytes(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        assert!(
            parsed.get("safety_identifier").is_none(),
            "safety_identifier is not nullable"
        );
    }

    #[test]
    fn output_config_maps_to_chat_generation_controls() {
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"output_config":{"effort":"high","format":{"type":"json_schema","schema":{"type":"object","properties":{"title":{"type":"string"}}}}},"messages":[{"role":"user","content":"Hi"}]}"#;
        let result = transform_bytes(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        assert_eq!(parsed["reasoning_effort"], "high", "effort mapped");
        assert_eq!(parsed["response_format"]["type"], "json_schema", "format mapped");
        assert_eq!(
            parsed["response_format"]["json_schema"]["schema"]["properties"]["title"]["type"], "string",
            "schema carried"
        );
        assert_eq!(
            parsed["response_format"]["json_schema"]["strict"], true,
            "Anthropic structured outputs guarantee conformance, so the Chat schema must be strict"
        );
        assert!(parsed.get("output_config").is_none(), "output_config must not travel");
    }

    #[test]
    fn chat_fields_whose_effect_the_response_cannot_carry_are_rejected() {
        for field in [
            "n",
            "logprobs",
            "top_logprobs",
            "audio",
            "modalities",
            "functions",
            "function_call",
            "web_search_options",
            "moderation",
        ] {
            let body = json!({
                "model": "claude-opus-4-8",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Hi"}],
                field: 2,
            });
            let error = transform_request(body).unwrap_err();
            assert!(
                error.contains(field),
                "rejection for `{field}` must name the field: {error}"
            );
        }
    }

    #[test]
    fn default_valued_rejected_fields_are_treated_as_absent() {
        for (field, default) in [
            ("n", json!(1)),
            ("logprobs", json!(false)),
            ("modalities", json!(["text"])),
            ("function_call", json!("none")),
            ("service_tier", json!("auto")),
            ("mcp_servers", json!([])),
        ] {
            let body = json!({
                "model": "claude-opus-4-8",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Hi"}],
                field: default,
            });
            let result = transform_request(body);
            assert!(
                result.is_ok(),
                "the default value of `{field}` must be accepted: {result:?}"
            );
            let parsed: Value = serde_json::from_slice(&result.unwrap()).unwrap();
            assert!(
                parsed.get(field).is_none(),
                "the default value of `{field}` must not be forwarded"
            );
        }
    }

    #[test]
    fn non_default_valued_rejected_fields_are_rejected() {
        for (field, value) in [
            ("n", json!(2)),
            ("logprobs", json!(true)),
            ("modalities", json!(["text", "audio"])),
            ("function_call", json!("auto")),
            ("service_tier", json!("standard_only")),
            (
                "mcp_servers",
                json!([{"type": "url", "url": "https://example.com", "name": "x"}]),
            ),
        ] {
            let body = json!({
                "model": "claude-opus-4-8",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Hi"}],
                field: value,
            });
            let error = transform_request(body).unwrap_err();
            assert!(
                error.contains(field),
                "a non-default `{field}` must be rejected by name: {error}"
            );
        }
    }

    #[test]
    fn output_config_null_values_are_treated_as_absent() {
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"output_config":{"effort":null,"format":null,"task_budget":null},"messages":[{"role":"user","content":"Hi"}]}"#;
        let result = transform_bytes(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        for field in ["reasoning_effort", "response_format", "output_config"] {
            assert!(
                parsed.get(field).is_none(),
                "a null `{field}` source must produce nothing"
            );
        }
    }

    #[test]
    fn unknown_null_output_config_key_is_rejected() {
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"output_config":{"effort":"high","foo":null},"messages":[{"role":"user","content":"Hi"}]}"#;
        let error = transform_bytes(body).unwrap_err();

        assert!(
            error.contains("output_config.foo"),
            "a null under an unknown key must still be rejected by name: {error}"
        );
    }

    #[test]
    fn unsupported_output_config_keys_are_rejected() {
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"output_config":{"effort":"high","task_budget":{"type":"tokens","budget":4096}},"messages":[{"role":"user","content":"Hi"}]}"#;
        let error = transform_bytes(body).unwrap_err();

        assert!(
            error.contains("output_config.task_budget"),
            "rejection must name the unsupported key: {error}"
        );
    }

    #[test]
    fn unsupported_output_format_type_is_rejected() {
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"output_config":{"format":{"type":"grammar"}},"messages":[{"role":"user","content":"Hi"}]}"#;
        let error = transform_bytes(body).unwrap_err();

        assert!(
            error.contains("output_config.format"),
            "rejection must name the unsupported format: {error}"
        );
    }

    #[test]
    fn deprecated_output_format_maps_to_response_format() {
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"output_format":{"type":"json_schema","schema":{"type":"object"}},"messages":[{"role":"user","content":"Hi"}]}"#;
        let result = transform_bytes(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        assert_eq!(parsed["response_format"]["type"], "json_schema", "format mapped");
        assert!(parsed.get("output_format").is_none(), "output_format must not travel");
    }

    #[test]
    fn non_null_unmappable_fields_are_rejected() {
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"thinking":{"type":"enabled","budget_tokens":1024},"context_management":{"edits":[{"type":"clear_thinking_20251015"}]},"messages":[{"role":"user","content":"Hi"}]}"#;
        let error = transform_bytes(body).unwrap_err();
        assert!(error.contains("thinking"), "{error}");
    }

    #[test]
    fn unrepresentable_fields_are_rejected() {
        for field in ["service_tier", "container", "inference_geo", "mcp_servers"] {
            let body = json!({
                "model": "claude-opus-4-8",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Hi"}],
                field: {"any": 1},
            });
            let error = transform_request(body).unwrap_err();
            assert!(
                error.contains(field),
                "rejection for `{field}` must name the field: {error}"
            );
        }
    }

    #[test]
    fn null_unrepresentable_field_is_treated_as_absent() {
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"service_tier":null,"messages":[{"role":"user","content":"Hi"}]}"#;
        let result = transform_bytes(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        assert!(parsed.get("service_tier").is_none(), "null field is not forwarded");
    }

    #[test]
    fn transform_request_non_json_body() {
        let body = b"not json at all";
        let result = transform_bytes(body);
        assert!(result.is_err(), "non-JSON body should return Err");
        assert!(
            result.unwrap_err().contains("invalid JSON"),
            "error should mention invalid JSON"
        );
    }

    #[test]
    fn transform_request_json_array_body() {
        let body = b"[1,2,3]";
        let result = transform_bytes(body);
        assert!(result.is_err(), "JSON array body should return Err");
        assert!(
            result.unwrap_err().contains("not a JSON object"),
            "error should mention not a JSON object"
        );
    }

    #[test]
    fn non_string_system_is_rejected() {
        let body =
            br#"{"model":"claude-opus-4-8","max_tokens":1024,"system":42,"messages":[{"role":"user","content":"Hi"}]}"#;
        let error = transform_bytes(body).unwrap_err();
        assert!(error.contains("`system`"), "{error}");
    }

    #[test]
    fn hoist_system_array_empty_text_skipped() {
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"system":[{"type":"text","text":""}],"messages":[{"role":"user","content":"Hi"}]}"#;
        let result = transform_bytes(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        assert_eq!(
            parsed["messages"].as_array().unwrap().len(),
            1,
            "system with single empty text block should be skipped"
        );
        assert_eq!(parsed["messages"][0]["role"], "user");
    }

    #[test]
    fn message_missing_role_is_rejected() {
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"messages":[{"content":"Hi"}]}"#;
        let error = transform_bytes(body).unwrap_err();
        assert!(error.contains("role"), "{error}");
    }

    #[test]
    fn message_non_text_content_is_rejected() {
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"messages":[{"role":"user","content":42}]}"#;
        let error = transform_bytes(body).unwrap_err();
        assert!(error.contains("content"), "{error}");
    }

    #[test]
    fn thinking_block_is_rejected() {
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"messages":[{"role":"assistant","content":[{"type":"thinking","thinking":"Let me think..."}]}]}"#;
        let error = transform_bytes(body).unwrap_err();
        assert!(error.contains("thinking"), "{error}");
    }

    #[test]
    fn unknown_block_type_is_rejected() {
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"messages":[{"role":"user","content":[{"type":"custom_xyz","data":"something"}]}]}"#;
        let error = transform_bytes(body).unwrap_err();
        assert!(error.contains("custom_xyz"), "{error}");
    }

    #[test]
    fn tool_choice_string_none() {
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"tool_choice":"none","messages":[{"role":"user","content":"Hi"}]}"#;
        let result = transform_bytes(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        assert_eq!(parsed["tool_choice"], "none", "string none maps to none");
    }

    #[test]
    fn unknown_string_tool_choice_is_rejected() {
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"tool_choice":"foo","messages":[{"role":"user","content":"Hi"}]}"#;
        let error = transform_bytes(body).unwrap_err();
        assert!(error.contains("`tool_choice`"), "{error}");
    }

    #[test]
    fn tool_choice_object_none() {
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"tools":[{"name":"f","description":"d","input_schema":{"type":"object","properties":{}}}],"tool_choice":{"type":"none"},"messages":[{"role":"user","content":"Hi"}]}"#;
        let result = transform_bytes(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        assert_eq!(parsed["tool_choice"], "none", "object-form none maps to none");
    }

    #[test]
    fn tool_choice_object_tool_with_name() {
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"tools":[{"name":"fn","description":"d","input_schema":{"type":"object","properties":{}}}],"tool_choice":{"type":"tool","name":"fn"},"messages":[{"role":"user","content":"Hi"}]}"#;
        let result = transform_bytes(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        assert_eq!(
            parsed["tool_choice"]["type"], "function",
            "tool type should map to function"
        );
        assert_eq!(
            parsed["tool_choice"]["function"]["name"], "fn",
            "tool name should be preserved"
        );
    }

    #[test]
    fn tool_choice_without_name_is_rejected() {
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"tools":[{"name":"f","description":"d","input_schema":{"type":"object","properties":{}}}],"tool_choice":{"type":"tool"},"messages":[{"role":"user","content":"Hi"}]}"#;
        let error = transform_bytes(body).unwrap_err();
        assert!(error.contains("`tool_choice`"), "{error}");
    }

    #[test]
    fn non_string_tool_choice_is_rejected() {
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"tool_choice":true,"messages":[{"role":"user","content":"Hi"}]}"#;
        let error = transform_bytes(body).unwrap_err();
        assert!(error.contains("`tool_choice`"), "{error}");
    }

    #[test]
    fn multipart_image_and_text_produces_array_content() {
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"messages":[{"role":"user","content":[{"type":"text","text":"Describe this"},{"type":"image","source":{"type":"url","url":"https://example.com/img.png"}}]}]}"#;
        let result = transform_bytes(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        let content = &parsed["messages"][0]["content"];
        assert!(content.is_array(), "multipart content should be an array");
        assert_eq!(content.as_array().unwrap().len(), 2, "two content parts");
        assert_eq!(content[0]["type"], "text");
        assert_eq!(content[1]["type"], "image_url");
    }

    #[test]
    fn two_text_blocks_stay_two_content_parts() {
        // String content is emitted only for a *single* text part. Two text
        // blocks keep their part boundaries rather than being joined.
        let body = br#"{"model":"m","messages":[{"role":"user","content":[{"type":"text","text":"one"},{"type":"text","text":"two"}]}]}"#;
        let result = transform_bytes(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        let content = parsed["messages"][0]["content"].as_array().unwrap();
        assert_eq!(content.len(), 2, "two text blocks stay two parts");
        assert_eq!(content[0]["text"], "one");
        assert_eq!(content[1]["text"], "two");
    }

    #[test]
    fn assistant_tool_calls_join_all_text_blocks() {
        // The assistant+tool_calls branch joins every text part into one string,
        // a different rule from the single-part case above.
        let body = br#"{"model":"m","messages":[{"role":"assistant","content":[{"type":"text","text":"one"},{"type":"text","text":"two"},{"type":"tool_use","id":"c1","name":"f","input":{}}]}]}"#;
        let result = transform_bytes(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        let msg = &parsed["messages"][0];
        assert_eq!(msg["content"], "onetwo", "assistant text parts are joined");
        assert_eq!(msg["tool_calls"][0]["id"], "c1");
    }

    #[test]
    fn assistant_tool_calls_distinguish_empty_text_from_no_text() {
        // An empty text block still emits `content: ""`; no text block at all
        // emits no `content` key.
        let empty = br#"{"model":"m","messages":[{"role":"assistant","content":[{"type":"text","text":""},{"type":"tool_use","id":"c1","name":"f","input":{}}]}]}"#;
        let parsed: Value = serde_json::from_slice(&transform_bytes(empty).unwrap()).unwrap();
        assert_eq!(
            parsed["messages"][0]["content"], "",
            "an empty text block still emits content"
        );

        let none = br#"{"model":"m","messages":[{"role":"assistant","content":[{"type":"tool_use","id":"c1","name":"f","input":{}}]}]}"#;
        let parsed: Value = serde_json::from_slice(&transform_bytes(none).unwrap()).unwrap();
        assert!(
            parsed["messages"][0].get("content").is_none(),
            "no text block emits no content key"
        );
    }

    #[test]
    fn assistant_tool_calls_with_image_are_rejected() {
        let body = br#"{"model":"m","messages":[{"role":"assistant","content":[{"type":"text","text":"one"},{"type":"image","source":{"type":"url","url":"https://example.com/i.png"}},{"type":"text","text":"two"},{"type":"tool_use","id":"c1","name":"f","input":{}}]}]}"#;
        let error = transform_bytes(body).unwrap_err();
        assert!(error.contains("image"), "{error}");
    }

    #[test]
    fn tool_use_without_input_serializes_an_empty_object() {
        let body = br#"{"model":"m","messages":[{"role":"assistant","content":[{"type":"tool_use","id":"c1","name":"f"},{"type":"tool_use","id":"c2","name":"g","input":null},{"type":"tool_use","id":"c3","name":"h","input":{}},{"type":"tool_use","id":"c4","name":"i","input":[1,2]},{"type":"tool_use","id":"c5","name":"j","input":"text"}]}]}"#;
        let result = transform_bytes(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        let calls = parsed["messages"][0]["tool_calls"].as_array().unwrap();
        assert_eq!(
            calls[0]["function"]["arguments"], "{}",
            "absent input becomes an empty object"
        );
        assert_eq!(
            calls[1]["function"]["arguments"], "null",
            "an explicit null input is preserved as null"
        );
        assert_eq!(calls[2]["function"]["arguments"], "{}");
        assert_eq!(calls[3]["function"]["arguments"], "[1,2]");
        assert_eq!(calls[4]["function"]["arguments"], "\"text\"");
    }

    #[test]
    fn only_tool_result_blocks_produce_tool_messages() {
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"call_1","content":"result1"},{"type":"tool_result","tool_use_id":"call_2","content":"result2"}]}]}"#;
        let result = transform_bytes(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        let messages = parsed["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 2, "two tool messages, no wrapper");
        assert_eq!(messages[0]["role"], "tool");
        assert_eq!(messages[0]["tool_call_id"], "call_1");
        assert_eq!(messages[0]["content"], "result1");
        assert_eq!(messages[1]["role"], "tool");
        assert_eq!(messages[1]["tool_call_id"], "call_2");
        assert_eq!(messages[1]["content"], "result2");
    }

    #[test]
    fn extract_tool_result_content_null() {
        let (text, images) = split_tool_result_content(Some(Value::Null));
        assert!(text.is_empty(), "null content should return empty string");
        assert!(images.is_empty(), "null content carries no images");
    }

    #[test]
    fn extract_tool_result_content_missing() {
        let (text, images) = split_tool_result_content(None);
        assert!(text.is_empty(), "missing content should return empty string");
        assert!(images.is_empty(), "missing content carries no images");
    }

    #[test]
    fn tool_result_content_split_keeps_text_and_images_in_one_pass() {
        let content = json!([
            {"type": "text", "text": "before"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}},
            {"type": "text", "text": "after"},
            {"type": "thinking", "thinking": "ignored"},
            "not an object"
        ]);

        let (text, images) = split_tool_result_content(Some(content));

        assert_eq!(text, "before\nafter", "text parts join in order, skipping non-text");
        assert_eq!(images.len(), 1, "one image part promoted");
        assert_eq!(images[0]["image_url"]["url"], "data:image/png;base64,AAAA");
    }

    #[test]
    fn server_tools_are_rejected() {
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"tools":[{"type":"bash_20241022","name":"bash"},{"type":"text_editor_20241022","name":"text_editor"},{"type":"code_execution_20250522","name":"code_execution"},{"type":"computer_20250124","name":"computer","display_width_px":1024,"display_height_px":768},{"type":"future_server_tool_20270101","name":"future_server_tool"},{"type":42,"name":"invalid_type","input_schema":{"type":"object"}},{"name":"get_weather","description":"Get weather","input_schema":{"type":"object","properties":{}}},{"type":"custom","name":"get_time","description":"Get time","input_schema":{"type":"object","properties":{}}}],"messages":[{"role":"user","content":"Hi"}]}"#;
        let error = transform_bytes(body).unwrap_err();
        assert!(error.contains("typed"), "{error}");
    }

    #[test]
    fn streaming_request_includes_usage_option() {
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"stream":true,"messages":[{"role":"user","content":"Hi"}]}"#;
        let result = transform_bytes(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        assert_eq!(parsed["stream"], true, "stream should be true");
        assert_eq!(
            parsed["stream_options"]["include_usage"], true,
            "stream_options.include_usage should be set"
        );
    }

    #[test]
    fn non_streaming_request_omits_stream_options() {
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"messages":[{"role":"user","content":"Hi"}]}"#;
        let result = transform_bytes(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        assert!(
            parsed.get("stream_options").is_none(),
            "stream_options should not be present without stream:true"
        );
    }

    #[test]
    fn stream_false_omits_stream_options() {
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"stream":false,"messages":[{"role":"user","content":"Hi"}]}"#;
        let result = transform_bytes(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        assert_eq!(parsed["stream"], false, "stream should be false");
        assert!(
            parsed.get("stream_options").is_none(),
            "stream_options should not be present when stream is false"
        );
    }

    #[test]
    fn empty_allowlist_matches_strict_wrapper() {
        // The public strict wrapper is degrading with an empty allowlist.
        let body = json!({
            "model": "m",
            "thinking": {"type": "enabled", "budget_tokens": 1024},
            "messages": [{"role": "user", "content": "Hi"}]
        });
        assert!(transform_request(body.clone()).is_err());
        assert!(transform_request_degrading(body, LossyFeatureAllowlist::default()).is_err());
    }

    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "asserts every Anthropic-structured cache_control position"
    )]
    fn prompt_caching_degradation_strips_markers_everywhere() {
        let body = json!({
            "model": "m",
            "system": [{"type": "text", "text": "Be brief", "cache_control": {"type": "ephemeral"}}],
            "tools": [{
                "name": "lookup",
                "input_schema": {"type": "object"},
                "cache_control": {"type": "ephemeral", "ttl": "1h"}
            }],
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "Hi", "cache_control": {"type": "ephemeral"}}
                ]},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "c1", "name": "lookup", "input": {"q": "x"}, "cache_control": {"type": "ephemeral"}}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "c1", "cache_control": {"type": "ephemeral"}, "content": [
                        {"type": "text", "text": "Sunny", "cache_control": {"type": "ephemeral"}}
                    ]}
                ]}
            ]
        });
        let (translated, degraded) = degrade(body, CACHE).unwrap();
        assert!(degraded.prompt_caching, "cache markers were removed");
        assert!(!degraded.extended_thinking);
        let serialized = translated.to_string();
        assert!(
            !serialized.contains("cache_control"),
            "no cache_control survives translation: {serialized}"
        );
        // Prompt and tool content is preserved. Top-level system is hoisted to a
        // leading system message, then the user/assistant/tool turns follow.
        assert_eq!(
            translated["messages"][0],
            json!({"role": "system", "content": "Be brief"})
        );
        assert_eq!(translated["messages"][1]["content"], "Hi");
        assert_eq!(translated["tools"][0]["function"]["name"], "lookup");
        assert_eq!(
            translated["messages"][2]["tool_calls"][0]["function"]["arguments"],
            "{\"q\":\"x\"}"
        );
    }

    #[test]
    fn prompt_caching_degradation_never_touches_tool_input_user_data() {
        // A cache_control key inside tool_use.input is user data, not an
        // Anthropic marker, and must survive verbatim as serialized arguments.
        let body = json!({
            "model": "m",
            "messages": [{"role": "assistant", "content": [
                {"type": "tool_use", "id": "c1", "name": "echo", "input": {"cache_control": {"type": "ephemeral"}, "x": 1}}
            ]}]
        });
        let (translated, degraded) = degrade(body, CACHE).unwrap();
        assert!(
            !degraded.prompt_caching,
            "no Anthropic marker was present, so nothing is degraded"
        );
        let args = translated["messages"][0]["tool_calls"][0]["function"]["arguments"]
            .as_str()
            .unwrap();
        let parsed_args: Value = serde_json::from_str(args).unwrap();
        assert_eq!(
            parsed_args["cache_control"],
            json!({"type": "ephemeral"}),
            "user data inside tool input is preserved: {args}"
        );
    }

    #[test]
    fn prompt_caching_degradation_rejects_malformed_marker() {
        for marker in [
            json!("ephemeral"),
            json!({"type": "forever"}),
            json!({"type": "ephemeral", "scope": "x"}),
            // Only the schema's `5m`/`1h` TTLs are recognized; an arbitrary
            // string or a non-string value fails closed rather than being dropped.
            json!({"type": "ephemeral", "ttl": "7m"}),
            json!({"type": "ephemeral", "ttl": 300}),
        ] {
            let body = json!({
                "model": "m",
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": "Hi", "cache_control": marker}
                ]}]
            });
            let error = transform_request_degrading(body, CACHE).unwrap_err();
            assert!(error.contains("cache_control"), "{error}");
        }
    }

    #[test]
    fn prompt_caching_not_allowlisted_still_rejects() {
        let body = json!({
            "model": "m",
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "Hi", "cache_control": {"type": "ephemeral"}}
            ]}]
        });
        assert!(transform_request_degrading(body, THINK).is_err());
    }

    #[test]
    fn prompt_caching_degradation_strips_top_level_cache_control() {
        // A top-level request `cache_control` is Anthropic cache metadata, not a
        // mapped field, so degradation strips it like any block-level marker.
        let body = json!({
            "model": "m",
            "cache_control": {"type": "ephemeral", "ttl": "1h"},
            "messages": [{"role": "user", "content": "Hi"}]
        });
        let (translated, degraded) = degrade(body, CACHE).unwrap();
        assert!(degraded.prompt_caching, "the top-level marker counts as degradation");
        assert!(
            translated.get("cache_control").is_none(),
            "the top-level marker is stripped: {translated}"
        );
    }

    #[test]
    fn top_level_cache_control_not_allowlisted_still_rejects() {
        // Without the allowlist the strict validator must fail closed rather than
        // forward the unmappable marker.
        let body = json!({
            "model": "m",
            "cache_control": {"type": "ephemeral"},
            "messages": [{"role": "user", "content": "Hi"}]
        });
        let error = transform_request_degrading(body, THINK).unwrap_err();
        assert!(error.contains("cache_control"), "{error}");
    }

    #[test]
    fn top_level_cache_control_malformed_marker_rejects() {
        // A present-but-malformed top-level marker fails closed even when caching
        // is allowlisted, so an unsupported shape is never silently dropped.
        let body = json!({
            "model": "m",
            "cache_control": {"type": "forever"},
            "messages": [{"role": "user", "content": "Hi"}]
        });
        let error = transform_request_degrading(body, CACHE).unwrap_err();
        assert!(error.contains("cache_control"), "{error}");
    }

    #[test]
    fn extended_thinking_degradation_accepts_adaptive_and_display() {
        // The `adaptive` shape and the optional `display` mode are valid thinking
        // requests per the Anthropic discriminator, so they degrade, not reject.
        for thinking in [
            json!({"type": "adaptive"}),
            json!({"type": "adaptive", "display": "omitted"}),
            json!({"type": "enabled", "budget_tokens": 1024, "display": "summarized"}),
        ] {
            let body = json!({
                "model": "m",
                "max_tokens": 2048,
                "thinking": thinking.clone(),
                "messages": [{"role": "user", "content": "Hi"}]
            });
            let (translated, degraded) = degrade(body, THINK).unwrap();
            assert!(degraded.extended_thinking, "{thinking} degrades");
            assert!(translated.get("thinking").is_none(), "thinking removed for {thinking}");
        }
    }

    #[test]
    fn extended_thinking_degradation_rejects_unrecognized_thinking_shapes() {
        // `enabled` requires an integer `budget_tokens` ≥1024, a `display` must be
        // a defined mode, and unknown keys are not a recognized shape.
        for thinking in [
            json!({"type": "enabled"}),
            json!({"type": "enabled", "budget_tokens": 512}),
            json!({"type": "adaptive", "display": "verbose"}),
            json!({"type": "adaptive", "budget_tokens": 1024}),
        ] {
            let body = json!({
                "model": "m",
                "max_tokens": 2048,
                "thinking": thinking.clone(),
                "messages": [{"role": "user", "content": "Hi"}]
            });
            let error = transform_request_degrading(body, THINK).unwrap_err();
            assert!(error.contains("thinking"), "{thinking} -> {error}");
        }
    }

    #[test]
    fn extended_thinking_degradation_rejects_budget_at_or_above_max_tokens() {
        // The schema requires `budget_tokens` to stay below `max_tokens`; an
        // at-or-above budget is malformed and must fail closed rather than be
        // stripped and reported as a clean degradation.
        for max_tokens in [1024, 2000, 2048] {
            let body = json!({
                "model": "m",
                "max_tokens": max_tokens,
                "thinking": {"type": "enabled", "budget_tokens": 2048},
                "messages": [{"role": "user", "content": "Hi"}]
            });
            let error = transform_request_degrading(body, THINK).unwrap_err();
            assert!(error.contains("max_tokens"), "max_tokens {max_tokens} -> {error}");
        }
    }

    #[test]
    fn extended_thinking_degradation_strips_thinking_and_edits() {
        let body = json!({
            "model": "m",
            "max_tokens": 2048,
            "thinking": {"type": "enabled", "budget_tokens": 1024},
            "context_management": {"edits": [{"type": "clear_thinking_20251015"}]},
            "messages": [{"role": "user", "content": "Hi"}]
        });
        let (translated, degraded) = degrade(body, THINK).unwrap();
        assert!(degraded.extended_thinking);
        assert!(!degraded.prompt_caching);
        assert!(translated.get("thinking").is_none(), "thinking removed");
        assert!(
            translated.get("context_management").is_none(),
            "thinking-only context_management removed"
        );
        assert_eq!(translated["messages"][0]["content"], "Hi");
    }

    #[test]
    fn extended_thinking_degradation_accepts_disabled_thinking() {
        let body = json!({
            "model": "m",
            "thinking": {"type": "disabled"},
            "messages": [{"role": "user", "content": "Hi"}]
        });
        let (_, degraded) = degrade(body, THINK).unwrap();
        assert!(degraded.extended_thinking);
    }

    #[test]
    fn extended_thinking_degradation_rejects_malformed_thinking() {
        for thinking in [
            json!("on"),
            json!({"type": "weird"}),
            json!({"type": "enabled", "budget_tokens": "lots"}),
        ] {
            let body = json!({
                "model": "m",
                "thinking": thinking,
                "messages": [{"role": "user", "content": "Hi"}]
            });
            let error = transform_request_degrading(body, THINK).unwrap_err();
            assert!(error.contains("thinking"), "{error}");
        }
    }

    #[test]
    fn extended_thinking_degradation_rejects_non_thinking_context_edits() {
        for cm in [
            json!({"edits": [{"type": "clear_tool_uses_20250919"}]}),
            json!({"edits": [{"type": "clear_thinking_20251015"}, {"type": "clear_tool_uses_20250919"}]}),
            // Only the exact `clear_thinking_20251015` edit is a thinking edit;
            // a look-alike version is an unrecognized edit, not a thinking one.
            json!({"edits": [{"type": "clear_thinking_20990101"}]}),
            json!({"trigger": "x"}),
            json!([]),
        ] {
            let body = json!({
                "model": "m",
                "context_management": cm,
                "messages": [{"role": "user", "content": "Hi"}]
            });
            let error = transform_request_degrading(body, THINK).unwrap_err();
            assert!(error.contains("context_management"), "{error}");
        }
    }

    #[test]
    fn extended_thinking_degradation_accepts_recognized_clear_thinking_keep() {
        // Every `keep` shape the Anthropic `ClearThinking20251015` schema defines
        // is a recognized thinking edit and degrades rather than rejecting.
        for keep in [
            json!("all"),
            json!({"type": "all"}),
            json!({"type": "thinking_turns", "value": 3}),
        ] {
            let body = json!({
                "model": "m",
                "max_tokens": 2048,
                "context_management": {"edits": [{"type": "clear_thinking_20251015", "keep": keep.clone()}]},
                "messages": [{"role": "user", "content": "Hi"}]
            });
            let (translated, degraded) = degrade(body, THINK).unwrap();
            assert!(degraded.extended_thinking, "keep {keep} degrades");
            assert!(
                translated.get("context_management").is_none(),
                "context_management removed for keep {keep}"
            );
        }
    }

    #[test]
    fn extended_thinking_degradation_accepts_empty_context_management() {
        // `edits` is optional (`minItems: 0`), so a context_management with no
        // edits — the empty object `{}` or an empty `edits` array — is a valid
        // no-op: it is stripped and served, not rejected, and removing a no-op is
        // not itself a degradation.
        for cm in [json!({}), json!({"edits": []})] {
            let body = json!({
                "model": "m",
                "max_tokens": 2048,
                "context_management": cm.clone(),
                "messages": [{"role": "user", "content": "Hi"}]
            });
            let (translated, degraded) = degrade(body, THINK).unwrap();
            assert!(
                !degraded.extended_thinking,
                "empty context_management {cm} is a no-op, not a degradation"
            );
            assert!(
                translated.get("context_management").is_none(),
                "empty context_management {cm} removed"
            );
        }
    }

    #[test]
    fn extended_thinking_degradation_rejects_malformed_clear_thinking_edits() {
        // Unknown edit fields and malformed `keep` values are rejected, not
        // stripped and reported as a clean degradation; the checked-in schema
        // rejects these shapes.
        for cm in [
            json!({"edits": [{"type": "clear_thinking_20251015", "trigger": {"type": "input_tokens", "value": 1}}]}),
            json!({"edits": [{"type": "clear_thinking_20251015", "keep": 3}]}),
            json!({"edits": [{"type": "clear_thinking_20251015", "keep": "recent"}]}),
            // The `keep` union carries no null, so explicit null is malformed and
            // must fail closed rather than be stripped as a clean degradation.
            json!({"edits": [{"type": "clear_thinking_20251015", "keep": null}]}),
            json!({"edits": [{"type": "clear_thinking_20251015", "keep": {"type": "thinking_turns"}}]}),
            json!({"edits": [{"type": "clear_thinking_20251015", "keep": {"type": "thinking_turns", "value": 0}}]}),
            json!({"edits": [{"type": "clear_thinking_20251015", "keep": {"type": "all", "value": 3}}]}),
            json!({"edits": [{"type": "clear_thinking_20251015", "keep": {"type": "nonsense"}}]}),
        ] {
            let body = json!({
                "model": "m",
                "max_tokens": 2048,
                "context_management": cm.clone(),
                "messages": [{"role": "user", "content": "Hi"}]
            });
            let error = transform_request_degrading(body, THINK).unwrap_err();
            assert!(error.contains("context_management"), "{cm} -> {error}");
        }
    }

    #[test]
    fn context_management_not_allowlisted_still_rejects() {
        let body = json!({
            "model": "m",
            "context_management": {"edits": [{"type": "clear_thinking_20251015"}]},
            "messages": [{"role": "user", "content": "Hi"}]
        });
        assert!(transform_request_degrading(body, CACHE).is_err());
    }

    #[test]
    fn degradation_reports_each_feature_independently() {
        let body = json!({
            "model": "m",
            "max_tokens": 2048,
            "thinking": {"type": "enabled", "budget_tokens": 1024},
            "system": [{"type": "text", "text": "Be brief", "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": "Hi"}]
        });
        let (_, degraded) = degrade(body, BOTH).unwrap();
        assert_eq!(
            degraded,
            DegradedFeatures {
                prompt_caching: true,
                extended_thinking: true
            }
        );
    }

    #[test]
    fn allowlisted_request_without_markers_reports_no_degradation() {
        let body = json!({
            "model": "m",
            "messages": [{"role": "user", "content": "Hi"}]
        });
        let (_, degraded) = degrade(body, BOTH).unwrap();
        assert!(!degraded.any(), "nothing to degrade");
    }

    // Test Utilities

    const CACHE: LossyFeatureAllowlist = LossyFeatureAllowlist {
        prompt_caching: true,
        extended_thinking: false,
    };
    const THINK: LossyFeatureAllowlist = LossyFeatureAllowlist {
        prompt_caching: false,
        extended_thinking: true,
    };
    const BOTH: LossyFeatureAllowlist = LossyFeatureAllowlist {
        prompt_caching: true,
        extended_thinking: true,
    };

    fn degrade(body: Value, allow: LossyFeatureAllowlist) -> Result<(Value, DegradedFeatures), String> {
        let output = transform_request_degrading(body, allow)?;
        let parsed = serde_json::from_slice(&output.body).map_err(|e| e.to_string())?;
        Ok((parsed, output.degraded))
    }
}
