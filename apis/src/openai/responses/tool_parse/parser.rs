// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Pure request body parser for tool definitions in the Responses API.
//!
//! Extracts tool types, function tool metadata, built-in tool
//! configurations, and `tool_choice` from a JSON request body.
//! No I/O, no side effects, no mutation of input bytes.

// -----------------------------------------------------------------------------
// ToolType
// -----------------------------------------------------------------------------

/// Classified tool type from the `tools` array.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ToolType {
    /// Client-defined function tool.
    Function,
    /// Built-in web search tool (includes all preview variants).
    WebSearch,
    /// Built-in file/vector-store search tool.
    FileSearch,
    /// Built-in code interpreter tool (sandboxed Python execution).
    CodeInterpreter,
    /// Built-in computer use tool (screen automation).
    ComputerUse,
    /// Built-in image generation tool (DALL-E).
    ImageGeneration,
    /// Built-in tool search/discovery tool.
    ToolSearch,
    /// MCP server tool.
    Mcp,
    /// Tool entry is missing a valid discriminator.
    Unclassified,
    /// Unrecognized tool type (forwarded to inference as-is).
    Unknown(String),
}

// -----------------------------------------------------------------------------
// FunctionTool
// -----------------------------------------------------------------------------

/// Extracted metadata from a function tool definition.
///
/// Only fields the proxy needs for routing and dispatch are extracted.
/// The full parameter schema is forwarded to inference untouched.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct FunctionTool {
    /// Tool name used for dispatch matching.
    pub name: String,
    /// Human-readable description (if present).
    pub description: Option<String>,
    /// Whether strict schema validation is requested.
    pub strict: Option<bool>,
}

// -----------------------------------------------------------------------------
// WebSearchConfig
// -----------------------------------------------------------------------------

/// Extracted configuration from a `web_search` tool entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct WebSearchConfig {
    /// Controls how much surrounding context to include with search
    /// results (`low`, `medium`, or `high`).
    pub search_context_size: Option<String>,
}

// -----------------------------------------------------------------------------
// FileSearchConfig
// -----------------------------------------------------------------------------

/// Extracted configuration from a `file_search` tool entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct FileSearchConfig {
    /// IDs of vector stores to search.
    pub vector_store_ids: Vec<String>,
    /// Maximum number of results per store.
    pub max_num_results: Option<u64>,
}

// -----------------------------------------------------------------------------
// ToolChoice
// -----------------------------------------------------------------------------

/// Parsed `tool_choice` value from the request.
///
/// Read to understand dispatch behavior but forwarded to inference
/// as-is. The proxy does not validate or normalize this value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ToolChoice {
    /// Automatic tool selection (default).
    Auto,
    /// No tool execution.
    None,
    /// Model must call at least one tool.
    Required,
    /// Model must call a specific function tool by name.
    Specific {
        /// Tool name to call.
        name: String,
    },
    /// Model is constrained to a set of allowed tools.
    Allowed {
        /// Tool selection mode for the allowed set (optional).
        mode: Option<String>,
    },
    /// Model must call a specific hosted tool type.
    Hosted {
        /// Hosted tool type to call.
        tool_type: String,
    },
    /// Model must call a specific MCP tool.
    Mcp {
        /// MCP server label.
        server_label: Option<String>,
        /// MCP tool name.
        name: Option<String>,
    },
    /// Unrecognized string value (forwarded as-is).
    Other(String),
}

impl ToolChoice {
    /// Stable string representation for metadata and filter results.
    pub(crate) fn as_str(&self) -> &str {
        match self {
            Self::Auto => "auto",
            Self::None => "none",
            Self::Required => "required",
            Self::Specific { name } => name,
            Self::Allowed { mode } => mode.as_deref().unwrap_or("allowed_tools"),
            Self::Hosted { tool_type } => tool_type,
            Self::Mcp { .. } => "mcp",
            Self::Other(s) => s,
        }
    }

    /// The `type` discriminator from the original object value.
    ///
    /// Returns `None` for string-valued choices (`auto`, `none`,
    /// `required`, unknown strings) since they have no object type.
    pub(crate) fn type_str(&self) -> Option<&str> {
        match self {
            Self::Auto | Self::None | Self::Required | Self::Other(_) => None,
            Self::Specific { .. } => Some("function"),
            Self::Allowed { .. } => Some("allowed_tools"),
            Self::Hosted { tool_type } => Some(tool_type),
            Self::Mcp { .. } => Some("mcp"),
        }
    }
}

// -----------------------------------------------------------------------------
// ParsedTools
// -----------------------------------------------------------------------------

/// Result of parsing the `tools` array and `tool_choice` from a
/// request body.
#[derive(Debug)]
#[expect(clippy::struct_excessive_bools, reason = "presence flags for each hosted tool type")]
pub(crate) struct ParsedTools {
    /// Extracted function tool definitions.
    pub function_tools: Vec<FunctionTool>,
    /// Web search configuration (if a `web_search` entry is present).
    pub web_search: Option<WebSearchConfig>,
    /// File search configuration (if a `file_search` entry is present).
    pub file_search: Option<FileSearchConfig>,
    /// Number of MCP tool entries found. Full payloads will be
    /// extracted when the MCP tool listing filter (#43) is added.
    ///
    /// Until then, only the count is needed for routing/presence.
    /// Parsed `tool_choice` value.
    pub tool_choice: Option<ToolChoice>,
    /// Number of function tools.
    pub function_count: usize,
    /// Number of built-in hosted tools (`web_search`, `file_search`,
    /// `code_interpreter`, `computer_use`, `image_generation`,
    /// `tool_search`).
    pub builtin_count: usize,
    /// Number of MCP tools.
    pub mcp_count: usize,
    /// Number of unknown tool types.
    pub unknown_count: usize,
    /// Whether a `code_interpreter` tool is present.
    pub has_code_interpreter: bool,
    /// Whether a `computer_use` tool is present.
    pub has_computer_use: bool,
    /// Whether an `image_generation` tool is present.
    pub has_image_generation: bool,
    /// Whether a `tool_search` tool is present.
    pub has_tool_search: bool,
}

impl ParsedTools {
    /// Build an empty result with no tools and no `tool_choice`.
    fn empty() -> Self {
        Self {
            function_tools: Vec::new(),
            web_search: None,
            file_search: None,
            tool_choice: None,
            function_count: 0,
            builtin_count: 0,
            mcp_count: 0,
            unknown_count: 0,
            has_code_interpreter: false,
            has_computer_use: false,
            has_image_generation: false,
            has_tool_search: false,
        }
    }

    /// Whether any recognised tool entries were found.
    ///
    /// Unlike the classifier's coarse `has_tools` (non-empty array
    /// check), this counts only entries whose `type` discriminator
    /// resolved to a known or unknown-but-typed variant. Entries
    /// missing a `type` field are ignored, so the two checks can
    /// disagree on malformed arrays.
    pub(crate) fn has_tools(&self) -> bool {
        self.function_count + self.builtin_count + self.mcp_count + self.unknown_count > 0
    }

    /// Whether a `web_search` tool is present.
    pub(crate) fn has_web_search(&self) -> bool {
        self.web_search.is_some()
    }

    /// Whether a `file_search` tool is present.
    pub(crate) fn has_file_search(&self) -> bool {
        self.file_search.is_some()
    }

    /// Whether any MCP tools are present.
    pub(crate) fn has_mcp(&self) -> bool {
        self.mcp_count > 0
    }

    /// Whether a `code_interpreter` tool is present.
    pub(crate) fn has_code_interpreter(&self) -> bool {
        self.has_code_interpreter
    }

    /// Whether a `computer_use` tool is present.
    pub(crate) fn has_computer_use(&self) -> bool {
        self.has_computer_use
    }

    /// Whether an `image_generation` tool is present.
    pub(crate) fn has_image_generation(&self) -> bool {
        self.has_image_generation
    }

    /// Whether a `tool_search` tool is present.
    pub(crate) fn has_tool_search(&self) -> bool {
        self.has_tool_search
    }
}

// -----------------------------------------------------------------------------
// parse_tools
// -----------------------------------------------------------------------------

/// Parse tool definitions and `tool_choice` from a JSON request body.
///
/// This function is pure: no I/O, no side effects. It extracts only
/// the fields the proxy needs for routing and dispatch. Unknown tool
/// types are counted but not rejected — they are forwarded to
/// inference as-is.
pub(crate) fn parse_tools(body: &[u8]) -> ParsedTools {
    if body.is_empty() {
        return ParsedTools::empty();
    }

    let Ok(value) = serde_json::from_slice::<serde_json::Value>(body) else {
        return ParsedTools::empty();
    };

    let Some(obj) = value.as_object() else {
        return ParsedTools::empty();
    };

    let tool_choice = parse_tool_choice(obj);

    let Some(tools_array) = obj.get("tools").and_then(serde_json::Value::as_array) else {
        return ParsedTools {
            tool_choice,
            ..ParsedTools::empty()
        };
    };

    classify_tools_array(tools_array, tool_choice)
}

/// Walk the tools array and classify each entry by type.
fn classify_tools_array(tools_array: &[serde_json::Value], tool_choice: Option<ToolChoice>) -> ParsedTools {
    let mut acc = ParsedTools::empty();
    acc.tool_choice = tool_choice;

    for entry in tools_array {
        let Some(entry_obj) = entry.as_object() else {
            continue;
        };
        accumulate_tool(&mut acc, entry_obj);
    }

    acc
}

/// Classify and accumulate a single tool entry.
///
/// Extracts routing-relevant details (names, configs) without
/// cloning opaque payloads. `tool_dispatch` (#26) will add
/// owned extraction when it needs full tool entries.
fn accumulate_tool(acc: &mut ParsedTools, entry_obj: &serde_json::Map<String, serde_json::Value>) {
    match classify_tool_type(entry_obj) {
        ToolType::Function => {
            acc.function_count += 1;
            if let Some(ft) = extract_function_tool(entry_obj) {
                acc.function_tools.push(ft);
            }
        },
        ToolType::WebSearch if acc.web_search.is_none() => {
            acc.web_search = Some(extract_web_search_config(entry_obj));
            acc.builtin_count += 1;
        },
        ToolType::FileSearch if acc.file_search.is_none() => {
            acc.file_search = Some(extract_file_search_config(entry_obj));
            acc.builtin_count += 1;
        },
        hosted @ (ToolType::CodeInterpreter
        | ToolType::ComputerUse
        | ToolType::ImageGeneration
        | ToolType::ToolSearch) => {
            accumulate_hosted_tool(acc, &hosted);
        },
        ToolType::Mcp => {
            acc.mcp_count += 1;
        },
        ToolType::Unknown(_) => {
            acc.unknown_count += 1;
        },
        ToolType::Unclassified | ToolType::WebSearch | ToolType::FileSearch => {},
    }
}

/// Accumulate a hosted tool type with first-wins dedup.
fn accumulate_hosted_tool(acc: &mut ParsedTools, tool_type: &ToolType) {
    let flag = match tool_type {
        ToolType::CodeInterpreter => &mut acc.has_code_interpreter,
        ToolType::ComputerUse => &mut acc.has_computer_use,
        ToolType::ImageGeneration => &mut acc.has_image_generation,
        ToolType::ToolSearch => &mut acc.has_tool_search,
        _ => return,
    };
    if !*flag {
        *flag = true;
        acc.builtin_count += 1;
    }
}

// -----------------------------------------------------------------------------
// Private Helpers
// -----------------------------------------------------------------------------

/// Classify a tool entry by its `type` field.
fn classify_tool_type(obj: &serde_json::Map<String, serde_json::Value>) -> ToolType {
    let Some(type_value) = obj.get("type") else {
        return ToolType::Unclassified;
    };

    let Some(type_str) = type_value.as_str() else {
        return ToolType::Unclassified;
    };

    match type_str {
        "function" => ToolType::Function,
        "web_search" | "web_search_preview" | "web_search_preview_2025_03_11" | "web_search_2025_08_26" => {
            ToolType::WebSearch
        },

        "file_search" => ToolType::FileSearch,
        "code_interpreter" => ToolType::CodeInterpreter,
        "computer" | "computer_use" | "computer_use_preview" => ToolType::ComputerUse,
        "image_generation" => ToolType::ImageGeneration,
        "tool_search" => ToolType::ToolSearch,
        "mcp" => ToolType::Mcp,
        other => ToolType::Unknown(other.to_owned()),
    }
}

/// Extract proxy-needed fields from a function tool definition.
fn extract_function_tool(obj: &serde_json::Map<String, serde_json::Value>) -> Option<FunctionTool> {
    let name = obj.get("name").and_then(serde_json::Value::as_str)?.to_owned();

    let description = obj
        .get("description")
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned);

    let strict = obj.get("strict").and_then(serde_json::Value::as_bool);

    Some(FunctionTool {
        name,
        description,
        strict,
    })
}

/// Extract configuration from a `web_search` tool entry.
fn extract_web_search_config(obj: &serde_json::Map<String, serde_json::Value>) -> WebSearchConfig {
    let search_context_size = obj
        .get("search_context_size")
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned);

    WebSearchConfig { search_context_size }
}

/// Extract configuration from a `file_search` tool entry.
fn extract_file_search_config(obj: &serde_json::Map<String, serde_json::Value>) -> FileSearchConfig {
    let vector_store_ids = obj
        .get("vector_store_ids")
        .and_then(serde_json::Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(serde_json::Value::as_str)
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default();

    let max_num_results = obj.get("max_num_results").and_then(serde_json::Value::as_u64);

    FileSearchConfig {
        vector_store_ids,
        max_num_results,
    }
}

/// Parse the `tool_choice` field from the request object.
///
/// Object values are dispatched by their `type` discriminator so that
/// MCP choices (`{"type":"mcp","server_label":"...","name":"..."}`)
/// are not misidentified as `Specific` by an early `name` check.
fn parse_tool_choice(obj: &serde_json::Map<String, serde_json::Value>) -> Option<ToolChoice> {
    let value = obj.get("tool_choice")?;

    if let Some(s) = value.as_str() {
        return Some(match s {
            "auto" => ToolChoice::Auto,
            "none" => ToolChoice::None,
            "required" => ToolChoice::Required,
            other => ToolChoice::Other(other.to_owned()),
        });
    }

    let choice_obj = value.as_object()?;
    let type_str = choice_obj.get("type").and_then(serde_json::Value::as_str)?;
    parse_object_tool_choice(choice_obj, type_str)
}

/// Dispatch an object-valued `tool_choice` by its `type` field.
fn parse_object_tool_choice(obj: &serde_json::Map<String, serde_json::Value>, type_str: &str) -> Option<ToolChoice> {
    match type_str {
        "allowed_tools" => Some(ToolChoice::Allowed {
            mode: obj.get("mode").and_then(serde_json::Value::as_str).map(str::to_owned),
        }),
        "function" => obj
            .get("name")
            .and_then(serde_json::Value::as_str)
            .map(|name| ToolChoice::Specific { name: name.to_owned() }),
        "mcp" => Some(ToolChoice::Mcp {
            server_label: obj
                .get("server_label")
                .and_then(serde_json::Value::as_str)
                .map(str::to_owned),
            name: obj.get("name").and_then(serde_json::Value::as_str).map(str::to_owned),
        }),
        _ => Some(ToolChoice::Hosted {
            tool_type: type_str.to_owned(),
        }),
    }
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

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
mod tests {
    use super::*;

    #[test]
    fn parse_function_tools() {
        let body = br#"{
            "model": "gpt-4.1",
            "input": "test",
            "tools": [
                {"type": "function", "name": "get_weather", "description": "Get weather", "strict": true},
                {"type": "function", "name": "send_email", "description": "Send an email"}
            ]
        }"#;

        let result = parse_tools(body);

        assert_eq!(result.function_count, 2, "should find 2 function tools");
        assert_eq!(result.function_tools[0].name, "get_weather", "first tool name");
        assert_eq!(
            result.function_tools[0].description.as_deref(),
            Some("Get weather"),
            "first tool description"
        );
        assert_eq!(result.function_tools[0].strict, Some(true), "first tool strict flag");
        assert_eq!(result.function_tools[1].name, "send_email", "second tool name");
        assert!(
            result.function_tools[1].strict.is_none(),
            "second tool should have no strict flag"
        );
        assert!(result.has_tools(), "should have tools");
        assert!(!result.has_web_search(), "should not have web_search");
        assert!(!result.has_file_search(), "should not have file_search");
    }

    #[test]
    fn function_tool_without_name_still_sets_has_tools() {
        let body = br#"{"input": "test", "tools": [{"type": "function"}]}"#;
        let result = parse_tools(body);

        assert!(result.has_tools(), "nameless function tool should still set has_tools");
        assert_eq!(result.function_count, 1, "should count the discriminator");
        assert!(
            result.function_tools.is_empty(),
            "metadata not extractable without name"
        );
    }

    #[test]
    fn parse_web_search_tool() {
        let body = br#"{
            "model": "gpt-4.1",
            "input": "test",
            "tools": [{"type": "web_search", "search_context_size": "high"}]
        }"#;

        let result = parse_tools(body);

        assert!(result.has_web_search(), "should detect web_search");
        assert_eq!(result.builtin_count, 1, "should count 1 built-in tool");
        assert_eq!(
            result
                .web_search
                .as_ref()
                .and_then(|w| w.search_context_size.as_deref()),
            Some("high"),
            "should extract search_context_size"
        );
    }

    #[test]
    fn parse_web_search_preview_variant() {
        let body = br#"{
            "input": "test",
            "tools": [{"type": "web_search_preview_2025_03_11"}]
        }"#;

        let result = parse_tools(body);

        assert!(
            result.has_web_search(),
            "web_search_preview variant should be recognized"
        );
    }

    #[test]
    fn parse_web_search_2025_08_26_variant() {
        let body = br#"{
            "input": "test",
            "tools": [{"type": "web_search_2025_08_26"}]
        }"#;

        let result = parse_tools(body);

        assert!(
            result.has_web_search(),
            "web_search_2025_08_26 variant should be recognized"
        );
    }

    #[test]
    fn parse_file_search_tool() {
        let body = br#"{
            "model": "gpt-4.1",
            "input": "test",
            "tools": [{
                "type": "file_search",
                "vector_store_ids": ["vs_abc", "vs_def"],
                "max_num_results": 20
            }]
        }"#;

        let result = parse_tools(body);

        assert!(result.has_file_search(), "should detect file_search");
        assert_eq!(result.builtin_count, 1, "should count 1 built-in tool");

        let fs = result.file_search.as_ref().expect("file_search should be present");
        assert_eq!(fs.vector_store_ids, vec!["vs_abc", "vs_def"], "store IDs");
        assert_eq!(fs.max_num_results, Some(20), "max_num_results");
    }

    #[test]
    fn parse_mcp_tools() {
        let body = br#"{
            "input": "test",
            "tools": [{
                "type": "mcp",
                "server_label": "weather",
                "server_url": "http://localhost:8001/mcp"
            }]
        }"#;

        let result = parse_tools(body);

        assert!(result.has_mcp(), "should detect MCP tools");
        assert_eq!(result.mcp_count, 1, "should count 1 MCP tool");
    }

    #[test]
    fn parse_mixed_tools() {
        let body = br#"{
            "input": "test",
            "tools": [
                {"type": "function", "name": "calc"},
                {"type": "web_search"},
                {"type": "file_search", "vector_store_ids": ["vs_1"]},
                {"type": "mcp", "server_label": "srv"}
            ],
            "tool_choice": "auto"
        }"#;

        let result = parse_tools(body);

        assert_eq!(result.function_count, 1, "function count");
        assert_eq!(result.builtin_count, 2, "builtin count");
        assert_eq!(result.mcp_count, 1, "mcp count");
        assert!(result.has_tools(), "should have tools");
        assert!(result.has_web_search(), "should have web_search");
        assert!(result.has_file_search(), "should have file_search");
        assert!(result.has_mcp(), "should have mcp");
    }

    #[test]
    fn parse_tool_choice_auto() {
        let body = br#"{"input": "test", "tool_choice": "auto"}"#;
        let result = parse_tools(body);

        assert_eq!(
            result.tool_choice,
            Some(ToolChoice::Auto),
            "should parse tool_choice auto"
        );
    }

    #[test]
    fn parse_tool_choice_none() {
        let body = br#"{"input": "test", "tool_choice": "none"}"#;
        let result = parse_tools(body);

        assert_eq!(
            result.tool_choice,
            Some(ToolChoice::None),
            "should parse tool_choice none"
        );
    }

    #[test]
    fn parse_tool_choice_required() {
        let body = br#"{"input": "test", "tool_choice": "required"}"#;
        let result = parse_tools(body);

        assert_eq!(
            result.tool_choice,
            Some(ToolChoice::Required),
            "should parse tool_choice required"
        );
    }

    #[test]
    fn parse_tool_choice_specific() {
        let body = br#"{"input": "test", "tool_choice": {"type": "function", "name": "get_weather"}}"#;
        let result = parse_tools(body);

        assert_eq!(
            result.tool_choice,
            Some(ToolChoice::Specific {
                name: "get_weather".to_owned()
            }),
            "should parse specific tool_choice"
        );
    }

    #[test]
    fn parse_tool_choice_hosted_tool_type() {
        let body = br#"{"input": "test", "tool_choice": {"type": "file_search"}}"#;
        let result = parse_tools(body);

        assert_eq!(
            result.tool_choice,
            Some(ToolChoice::Hosted {
                tool_type: "file_search".to_owned()
            }),
            "hosted tool_choice should use type when name is absent"
        );
    }

    #[test]
    fn parse_tool_choice_allowed_tools_required() {
        let body = br#"{
            "input": "test",
            "tool_choice": {
                "type": "allowed_tools",
                "mode": "required",
                "tools": [{"type": "function", "name": "calc"}]
            }
        }"#;
        let result = parse_tools(body);

        assert_eq!(
            result.tool_choice,
            Some(ToolChoice::Allowed {
                mode: Some("required".to_owned())
            }),
            "allowed_tools tool_choice should use mode for routing"
        );
    }

    #[test]
    fn parse_tool_choice_allowed_tools_missing_mode() {
        let body = br#"{
            "input": "test",
            "tool_choice": {
                "type": "allowed_tools",
                "tools": [{"type": "function", "name": "calc"}]
            }
        }"#;
        let result = parse_tools(body);

        assert_eq!(
            result.tool_choice,
            Some(ToolChoice::Allowed { mode: None }),
            "allowed_tools without mode should still parse"
        );
    }

    #[test]
    fn parse_tool_choice_web_search_preview_type() {
        let body = br#"{"input": "test", "tool_choice": {"type": "web_search_preview"}}"#;
        let result = parse_tools(body);

        assert_eq!(
            result.tool_choice,
            Some(ToolChoice::Hosted {
                tool_type: "web_search_preview".to_owned()
            }),
            "web_search_preview tool_choice should use hosted tool type"
        );
    }

    #[test]
    fn parse_tool_choice_absent() {
        let body = br#"{"input": "test"}"#;
        let result = parse_tools(body);

        assert!(result.tool_choice.is_none(), "absent tool_choice should be None");
    }

    #[test]
    fn parse_tool_choice_unknown_string() {
        let body = br#"{"input": "test", "tool_choice": "custom_mode"}"#;
        let result = parse_tools(body);

        assert_eq!(
            result.tool_choice,
            Some(ToolChoice::Other("custom_mode".to_owned())),
            "unknown string tool_choice should be Other"
        );
    }

    #[test]
    fn empty_tools_array() {
        let body = br#"{"input": "test", "tools": []}"#;
        let result = parse_tools(body);

        assert!(!result.has_tools(), "empty tools array means no tools");
        assert_eq!(result.function_count, 0, "no function tools");
    }

    #[test]
    fn no_tools_field() {
        let body = br#"{"input": "test", "model": "gpt-4.1"}"#;
        let result = parse_tools(body);

        assert!(!result.has_tools(), "missing tools field means no tools");
    }

    #[test]
    fn tools_not_array() {
        let body = br#"{"input": "test", "tools": "invalid"}"#;
        let result = parse_tools(body);

        assert!(!result.has_tools(), "non-array tools should produce no tools");
    }

    #[test]
    fn parse_code_interpreter_tool() {
        let body = br#"{"input": "test", "tools": [{"type": "code_interpreter"}]}"#;
        let result = parse_tools(body);

        assert!(result.has_code_interpreter(), "should detect code_interpreter");
        assert_eq!(result.builtin_count, 1, "should count 1 built-in tool");
        assert!(result.has_tools(), "code_interpreter counts as has_tools");
    }

    #[test]
    fn parse_computer_use_tool() {
        let body = br#"{"input": "test", "tools": [{"type": "computer_use"}]}"#;
        let result = parse_tools(body);

        assert!(result.has_computer_use(), "should detect computer_use");
        assert_eq!(result.builtin_count, 1, "should count 1 built-in tool");
    }

    #[test]
    fn parse_computer_variant() {
        let body = br#"{"input": "test", "tools": [{"type": "computer"}]}"#;
        let result = parse_tools(body);

        assert!(
            result.has_computer_use(),
            "computer variant should be recognized as ComputerUse"
        );
    }

    #[test]
    fn parse_computer_use_preview_variant() {
        let body = br#"{"input": "test", "tools": [{"type": "computer_use_preview"}]}"#;
        let result = parse_tools(body);

        assert!(
            result.has_computer_use(),
            "computer_use_preview variant should be recognized"
        );
    }

    #[test]
    fn parse_image_generation_tool() {
        let body = br#"{"input": "test", "tools": [{"type": "image_generation"}]}"#;
        let result = parse_tools(body);

        assert!(result.has_image_generation(), "should detect image_generation");
        assert_eq!(result.builtin_count, 1, "should count 1 built-in tool");
    }

    #[test]
    fn parse_tool_search_tool() {
        let body = br#"{"input": "test", "tools": [{"type": "tool_search"}]}"#;
        let result = parse_tools(body);

        assert!(result.has_tool_search(), "should detect tool_search");
        assert_eq!(result.builtin_count, 1, "should count 1 built-in tool");
    }

    #[test]
    fn duplicate_code_interpreter_counted_once() {
        let body = br#"{
            "input": "test",
            "tools": [
                {"type": "code_interpreter"},
                {"type": "code_interpreter"}
            ]
        }"#;
        let result = parse_tools(body);

        assert_eq!(result.builtin_count, 1, "duplicate code_interpreter counted once");
    }

    #[test]
    fn unknown_tool_type() {
        let body = br#"{"input": "test", "tools": [{"type": "custom_tool"}]}"#;
        let result = parse_tools(body);

        assert_eq!(result.unknown_count, 1, "should count unknown tool type");
        assert!(result.has_tools(), "unknown tools still count as tools");
    }

    #[test]
    fn function_tool_missing_name() {
        let body = br#"{"input": "test", "tools": [{"type": "function", "description": "no name"}]}"#;
        let result = parse_tools(body);

        assert_eq!(
            result.function_count, 1,
            "discriminator should be counted even without name"
        );
        assert!(
            result.function_tools.is_empty(),
            "metadata not extractable without name"
        );
        assert!(result.has_tools(), "nameless function tool should still set has_tools");
    }

    #[test]
    fn function_tool_without_type_is_unclassified() {
        let body = br#"{"input": "test", "tools": [{"name": "calc"}]}"#;
        let result = parse_tools(body);

        assert_eq!(
            result.function_count, 0,
            "tool without type should not count as function"
        );
        assert!(!result.has_tools(), "tool without type should not set has_tools");
    }

    #[test]
    fn tool_entry_with_non_string_type_is_unclassified() {
        let body = br#"{"input": "test", "tools": [{"type": 123, "name": "calc"}]}"#;
        let result = parse_tools(body);

        assert_eq!(result.function_count, 0, "non-string type should not count as function");
        assert!(!result.has_tools(), "non-string type should not set has_tools");
    }

    #[test]
    fn empty_body() {
        let result = parse_tools(b"");

        assert!(!result.has_tools(), "empty body means no tools");
        assert!(result.tool_choice.is_none(), "no tool_choice from empty body");
    }

    #[test]
    fn invalid_json() {
        let result = parse_tools(b"not json");

        assert!(!result.has_tools(), "invalid JSON means no tools");
    }

    #[test]
    fn json_array_body() {
        let result = parse_tools(b"[1, 2, 3]");

        assert!(!result.has_tools(), "JSON array body should produce no tools");
    }

    #[test]
    fn non_object_tool_entry_skipped() {
        let body = br#"{"input": "test", "tools": ["not_an_object", 42]}"#;
        let result = parse_tools(body);

        assert!(!result.has_tools(), "non-object entries should be skipped");
    }

    #[test]
    fn parse_tool_choice_mcp() {
        let body = br#"{
            "input": "test",
            "tool_choice": {"type": "mcp", "server_label": "deepwiki", "name": "search"}
        }"#;
        let result = parse_tools(body);

        assert_eq!(
            result.tool_choice,
            Some(ToolChoice::Mcp {
                server_label: Some("deepwiki".to_owned()),
                name: Some("search".to_owned()),
            }),
            "MCP tool_choice should be parsed as Mcp variant"
        );
    }

    #[test]
    fn mcp_tool_choice_not_misidentified_as_specific() {
        let body = br#"{
            "input": "test",
            "tool_choice": {"type": "mcp", "server_label": "deepwiki", "name": "search"}
        }"#;
        let result = parse_tools(body);

        assert!(
            !matches!(result.tool_choice, Some(ToolChoice::Specific { .. })),
            "MCP tool_choice must not be parsed as Specific"
        );
        assert_eq!(
            result.tool_choice.as_ref().map(ToolChoice::as_str),
            Some("mcp"),
            "MCP tool_choice should promote as 'mcp'"
        );
    }

    #[test]
    fn parse_tool_choice_mcp_minimal() {
        let body = br#"{"input": "test", "tool_choice": {"type": "mcp"}}"#;
        let result = parse_tools(body);

        assert_eq!(
            result.tool_choice,
            Some(ToolChoice::Mcp {
                server_label: None,
                name: None,
            }),
            "MCP tool_choice without server_label/name should still parse"
        );
    }

    #[test]
    fn tool_choice_as_str_representation() {
        assert_eq!(ToolChoice::Auto.as_str(), "auto", "auto as_str");
        assert_eq!(ToolChoice::None.as_str(), "none", "none as_str");
        assert_eq!(ToolChoice::Required.as_str(), "required", "required as_str");
        assert_eq!(
            ToolChoice::Specific {
                name: "calc".to_owned()
            }
            .as_str(),
            "calc",
            "specific as_str"
        );
        assert_eq!(
            ToolChoice::Allowed {
                mode: Some("required".to_owned())
            }
            .as_str(),
            "required",
            "allowed_tools as_str should use mode"
        );
        assert_eq!(
            ToolChoice::Allowed { mode: None }.as_str(),
            "allowed_tools",
            "allowed_tools without mode as_str should fall back to type name"
        );
        assert_eq!(
            ToolChoice::Hosted {
                tool_type: "file_search".to_owned()
            }
            .as_str(),
            "file_search",
            "hosted as_str"
        );
        assert_eq!(
            ToolChoice::Mcp {
                server_label: Some("srv".to_owned()),
                name: Some("tool".to_owned()),
            }
            .as_str(),
            "mcp",
            "mcp as_str"
        );
        assert_eq!(ToolChoice::Other("x".to_owned()).as_str(), "x", "other as_str");
    }

    #[test]
    fn tool_choice_type_str_representation() {
        assert!(ToolChoice::Auto.type_str().is_none(), "auto has no type");
        assert!(ToolChoice::None.type_str().is_none(), "none has no type");
        assert!(ToolChoice::Required.type_str().is_none(), "required has no type");
        assert_eq!(
            ToolChoice::Specific {
                name: "calc".to_owned()
            }
            .type_str(),
            Some("function"),
            "specific type_str"
        );
        assert_eq!(
            ToolChoice::Allowed {
                mode: Some("required".to_owned())
            }
            .type_str(),
            Some("allowed_tools"),
            "allowed type_str"
        );
        assert_eq!(
            ToolChoice::Hosted {
                tool_type: "file_search".to_owned()
            }
            .type_str(),
            Some("file_search"),
            "hosted type_str"
        );
        assert_eq!(
            ToolChoice::Mcp {
                server_label: None,
                name: None,
            }
            .type_str(),
            Some("mcp"),
            "mcp type_str"
        );
        assert!(
            ToolChoice::Other("x".to_owned()).type_str().is_none(),
            "other has no type"
        );
    }

    #[test]
    fn duplicate_web_search_takes_first() {
        let body = br#"{
            "input": "test",
            "tools": [
                {"type": "web_search", "search_context_size": "high"},
                {"type": "web_search", "search_context_size": "low"}
            ]
        }"#;

        let result = parse_tools(body);

        assert_eq!(result.builtin_count, 1, "duplicate web_search counted once");
        assert_eq!(
            result
                .web_search
                .as_ref()
                .and_then(|w| w.search_context_size.as_deref()),
            Some("high"),
            "should take first web_search entry"
        );
    }

    #[test]
    fn file_search_empty_vector_store_ids() {
        let body = br#"{"input": "test", "tools": [{"type": "file_search"}]}"#;
        let result = parse_tools(body);

        assert!(result.has_file_search(), "file_search should be detected");
        assert!(
            result
                .file_search
                .as_ref()
                .expect("file_search present")
                .vector_store_ids
                .is_empty(),
            "missing vector_store_ids should default to empty"
        );
    }
}
