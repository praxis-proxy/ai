// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Load and import stored-session replay fixtures for integration tests.
//!
//! A replay fixture captures one or more sanitized turns from an agent
//! session and replays those turns through an example configuration.

use std::{
    fmt,
    path::{Component, PathBuf},
    str::FromStr,
};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Number, Value};

/// Stored-session protocol represented by a replay fixture.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ReplayProtocol {
    /// Anthropic Messages API traffic.
    AnthropicMessages,
    /// OpenAI Responses API traffic.
    OpenaiResponses,
}

/// A stored-session replay fixture loaded from JSON.
#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SessionReplay {
    /// Human-readable description of where the replay sample came from.
    pub source: String,
    /// API protocol used by all turns in the fixture.
    pub protocol: ReplayProtocol,
    /// Ordered request/response turns in the session.
    pub turns: Vec<ReplayTurn>,
}

/// A single request/response turn in a stored-session replay fixture.
#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ReplayTurn {
    /// Stable fixture-local turn name.
    pub name: String,
    /// HTTP path to replay this turn against.
    pub path: String,
    /// Request body sent by the agent.
    pub request: Value,
    /// JSON response body returned by the upstream model service.
    pub response: Value,
}

/// Raw session log content plus optional source metadata.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SessionInput<'a> {
    /// Raw session log content.
    content: &'a str,
    /// Optional human-readable source name, usually the input path.
    source_name: Option<&'a str>,
}

/// Session provider family recognized by replay importers.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SessionProvider {
    /// Claude Code JSONL session logs.
    ClaudeCode,
    /// Codex JSONL session logs.
    Codex,
}

/// Provider hint supplied by callers.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ProviderHint {
    /// Detect the provider from the session envelope.
    #[default]
    Auto,
    /// Force Claude Code import.
    ClaudeCode,
    /// Force Codex import.
    Codex,
}

/// Detection result from an importer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Detection {
    /// The importer recognized the input as this provider.
    Detected(SessionProvider),
    /// The importer did not recognize the input.
    NotDetected,
}

/// Options for converting raw session logs into replay fixtures.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ImportOptions {
    /// Provider selection strategy.
    pub provider: ProviderHint,
    /// Fallback `max_tokens` for Claude Code transcript imports.
    pub default_claude_max_tokens: u32,
}

/// Error returned while importing session logs.
#[derive(Debug, thiserror::Error)]
pub enum ImportError {
    /// A provider hint string was not recognized.
    #[error("unknown provider hint {hint:?}; expected auto, claude, claude_code, or codex")]
    UnknownProviderHint {
        /// User-provided provider hint.
        hint: String,
    },
    /// No importer recognized the session envelope.
    #[error("session log was not recognized as Claude Code or Codex")]
    UnrecognizedProvider,
    /// More than one importer recognized the session envelope.
    #[error("session log matched multiple providers")]
    AmbiguousProvider,
    /// The session matched a provider but did not contain importable turns.
    #[error("{provider} session did not contain importable request/response turns")]
    UnsupportedShape {
        /// Provider that recognized the session.
        provider: SessionProvider,
    },
    /// A selected importer found malformed JSONL.
    #[error("line {line}: invalid JSON: {source}")]
    InvalidJson {
        /// 1-based JSONL line number.
        line: usize,
        /// JSON parser error.
        #[source]
        source: serde_json::Error,
    },
}

/// Convert one provider-specific session log into a replay fixture.
pub trait SessionReplayImporter {
    /// Return the provider handled by this importer.
    fn provider(&self) -> SessionProvider;

    /// Detect whether this importer recognizes the input envelope.
    fn detect(&self, input: &SessionInput<'_>) -> Detection;

    /// Import an input session into the replay fixture schema.
    ///
    /// # Errors
    ///
    /// Returns [`ImportError`] when the input is malformed or does not contain
    /// turns this importer can convert safely.
    fn import(&self, input: &SessionInput<'_>, options: ImportOptions) -> Result<SessionReplay, ImportError>;
}

/// Importer for Codex session logs.
#[derive(Debug, Default)]
pub struct CodexSessionImporter;

/// Importer for Claude Code session logs.
#[derive(Debug, Default)]
pub struct ClaudeCodeSessionImporter;

/// Shared Codex importer instance for auto-detection.
static CODEX_IMPORTER: CodexSessionImporter = CodexSessionImporter;

/// Shared Claude Code importer instance for auto-detection.
static CLAUDE_CODE_IMPORTER: ClaudeCodeSessionImporter = ClaudeCodeSessionImporter;

impl<'a> SessionInput<'a> {
    /// Create input from raw session content.
    #[must_use]
    pub const fn new(content: &'a str) -> Self {
        Self {
            content,
            source_name: None,
        }
    }

    /// Attach a human-readable source name.
    #[must_use]
    pub const fn with_source_name(mut self, source_name: &'a str) -> Self {
        self.source_name = Some(source_name);
        self
    }

    /// Return the raw session content.
    #[must_use]
    pub const fn content(&self) -> &'a str {
        self.content
    }

    /// Return the optional source name.
    #[must_use]
    pub const fn source_name(&self) -> Option<&'a str> {
        self.source_name
    }
}

impl fmt::Display for SessionProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::ClaudeCode => "claude_code",
            Self::Codex => "codex",
        })
    }
}

impl FromStr for ProviderHint {
    type Err = ImportError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "auto" => Ok(Self::Auto),
            "claude" | "claude_code" | "claude-code" => Ok(Self::ClaudeCode),
            "codex" => Ok(Self::Codex),
            hint => Err(ImportError::UnknownProviderHint { hint: hint.to_owned() }),
        }
    }
}

impl Default for ImportOptions {
    fn default() -> Self {
        Self {
            provider: ProviderHint::Auto,
            default_claude_max_tokens: 1024,
        }
    }
}

impl SessionReplayImporter for CodexSessionImporter {
    fn provider(&self) -> SessionProvider {
        SessionProvider::Codex
    }

    fn detect(&self, input: &SessionInput<'_>) -> Detection {
        if input
            .content()
            .lines()
            .filter_map(parse_jsonl_line_lossy)
            .any(|value| is_codex_record(&value))
        {
            Detection::Detected(self.provider())
        } else {
            Detection::NotDetected
        }
    }

    fn import(&self, input: &SessionInput<'_>, _options: ImportOptions) -> Result<SessionReplay, ImportError> {
        let mut turns = Vec::new();

        for (line_index, line) in input.content().lines().enumerate() {
            let line_number = line_index + 1;
            let mut record = parse_jsonl_line(line, line_number)?;
            if let Some(turn) = codex_turn_from_record(&mut record, turns.len() + 1) {
                turns.push(turn);
            }
        }

        replay_from_turns(
            input,
            self.provider(),
            "Converted Codex session log",
            ReplayProtocol::OpenaiResponses,
            turns,
        )
    }
}

impl SessionReplayImporter for ClaudeCodeSessionImporter {
    fn provider(&self) -> SessionProvider {
        SessionProvider::ClaudeCode
    }

    fn detect(&self, input: &SessionInput<'_>) -> Detection {
        if input
            .content()
            .lines()
            .filter_map(parse_jsonl_line_lossy)
            .any(|value| is_claude_code_record(&value))
        {
            Detection::Detected(self.provider())
        } else {
            Detection::NotDetected
        }
    }

    fn import(&self, input: &SessionInput<'_>, options: ImportOptions) -> Result<SessionReplay, ImportError> {
        let mut turns = Vec::new();
        let mut last_user_message = None;

        for (line_index, line) in input.content().lines().enumerate() {
            let line_number = line_index + 1;
            let mut record = parse_jsonl_line(line, line_number)?;
            if let Some(user_message) = claude_user_message_from_record(&mut record) {
                last_user_message = Some(user_message);
                continue;
            }

            if !is_claude_assistant_record(&record) {
                continue;
            }
            let Some(user_message) = last_user_message.take() else {
                continue;
            };
            if let Some(turn) = claude_turn_from_record(&mut record, user_message, turns.len() + 1, options) {
                turns.push(turn);
            }
        }

        replay_from_turns(
            input,
            self.provider(),
            "Converted Claude Code session log",
            ReplayProtocol::AnthropicMessages,
            turns,
        )
    }
}

/// Import a session replay using either a provider hint or auto-detection.
///
/// # Errors
///
/// Returns [`ImportError`] when no provider is recognized, multiple providers
/// match, JSONL parsing fails, or the selected importer cannot extract safe
/// request/response turns.
pub fn import_session_replay(input: &SessionInput<'_>, options: ImportOptions) -> Result<SessionReplay, ImportError> {
    match options.provider {
        ProviderHint::Auto => import_with_auto_detection(input, options),
        ProviderHint::ClaudeCode => CLAUDE_CODE_IMPORTER.import(input, options),
        ProviderHint::Codex => CODEX_IMPORTER.import(input, options),
    }
}

/// Import by scanning all registered importers and selecting one match.
fn import_with_auto_detection(input: &SessionInput<'_>, options: ImportOptions) -> Result<SessionReplay, ImportError> {
    let mut matched = None;

    for importer in importers() {
        if matches!(importer.detect(input), Detection::Detected(_)) {
            if matched.is_some() {
                return Err(ImportError::AmbiguousProvider);
            }
            matched = Some(importer);
        }
    }

    let Some(importer) = matched else {
        return Err(ImportError::UnrecognizedProvider);
    };
    importer.import(input, options)
}

/// Return the built-in importer registry.
fn importers() -> [&'static dyn SessionReplayImporter; 2] {
    [&CODEX_IMPORTER, &CLAUDE_CODE_IMPORTER]
}

/// Parse a JSONL line for detection, ignoring malformed records.
fn parse_jsonl_line_lossy(line: &str) -> Option<Value> {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return None;
    }
    serde_json::from_str(trimmed).ok()
}

/// Parse a JSONL line for an active import.
fn parse_jsonl_line(line: &str, line_number: usize) -> Result<Value, ImportError> {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return Ok(Value::Null);
    }
    serde_json::from_str(trimmed).map_err(|source| ImportError::InvalidJson {
        line: line_number,
        source,
    })
}

/// Return true when a JSON object has the Codex session envelope.
fn is_codex_record(value: &Value) -> bool {
    value.as_object().is_some_and(|object| {
        object.contains_key("payload")
            && matches!(
                object.get("type").and_then(Value::as_str),
                Some("event_msg" | "response_item" | "session_meta" | "turn_context")
            )
    })
}

/// Return true when a JSON object has the Claude Code session envelope.
fn is_claude_code_record(value: &Value) -> bool {
    value.as_object().is_some_and(|object| {
        object.contains_key("uuid")
            && object.contains_key("sessionId")
            && object.contains_key("message")
            && matches!(object.get("type").and_then(Value::as_str), Some("assistant" | "user"))
    })
}

/// Extract an importable Codex turn from a JSONL record.
fn codex_turn_from_record(record: &mut Value, turn_number: usize) -> Option<ReplayTurn> {
    let record_object = record.as_object_mut()?;
    if record_object.get("type").and_then(Value::as_str) != Some("response_item") {
        return None;
    }
    let mut payload = record_object.remove("payload")?;
    let payload_object = payload.as_object_mut()?;
    let request = payload_object.remove("request")?;
    let response = payload_object.remove("response")?;

    Some(ReplayTurn {
        name: replay_turn_name("codex-turn", turn_number),
        path: "/v1/responses".to_owned(),
        request,
        response,
    })
}

/// Extract a Claude Code user message from a JSONL record.
fn claude_user_message_from_record(record: &mut Value) -> Option<Value> {
    let record_object = record.as_object_mut()?;
    if record_object.get("type").and_then(Value::as_str) != Some("user") {
        return None;
    }
    let message = record_object.remove("message")?;
    is_claude_user_message(&message).then_some(message)
}

/// Return true when a record is a Claude Code assistant record.
fn is_claude_assistant_record(record: &Value) -> bool {
    record.as_object().is_some_and(|object| {
        object.get("type").and_then(Value::as_str) == Some("assistant") && object.contains_key("message")
    })
}

/// Extract an importable Claude Code turn from a JSONL assistant record.
fn claude_turn_from_record(
    record: &mut Value,
    user_message: Value,
    turn_number: usize,
    options: ImportOptions,
) -> Option<ReplayTurn> {
    let record_object = record.as_object_mut()?;
    let message = record_object.remove("message")?;
    if !is_anthropic_assistant_message(&message) {
        return None;
    }
    let request = claude_request_from_response(&message, user_message, options)?;

    Some(ReplayTurn {
        name: replay_turn_name("claude-code-turn", turn_number),
        path: "/v1/messages".to_owned(),
        request,
        response: message,
    })
}

/// Return true when a Claude Code message is user-authored.
fn is_claude_user_message(message: &Value) -> bool {
    message.get("role").and_then(Value::as_str) == Some("user")
}

/// Return true when a Claude Code assistant message is an Anthropic response.
fn is_anthropic_assistant_message(message: &Value) -> bool {
    message.get("type").and_then(Value::as_str) == Some("message")
        && message.get("role").and_then(Value::as_str) == Some("assistant")
}

/// Build an Anthropic Messages request from a Claude Code transcript pair.
fn claude_request_from_response(response: &Value, user_message: Value, options: ImportOptions) -> Option<Value> {
    let model = response.get("model").and_then(Value::as_str)?;
    let mut request = Map::new();
    request.insert("model".to_owned(), Value::String(model.to_owned()));
    request.insert(
        "max_tokens".to_owned(),
        Value::Number(Number::from(options.default_claude_max_tokens)),
    );
    request.insert("messages".to_owned(), Value::Array(vec![user_message]));
    Some(Value::Object(request))
}

/// Build a stable generated turn name.
fn replay_turn_name(prefix: &str, number: usize) -> String {
    format!("{prefix}-{number}")
}

/// Build a fixture or return a provider-specific unsupported-shape error.
fn replay_from_turns(
    input: &SessionInput<'_>,
    provider: SessionProvider,
    fallback_source: &str,
    protocol: ReplayProtocol,
    turns: Vec<ReplayTurn>,
) -> Result<SessionReplay, ImportError> {
    if turns.is_empty() {
        return Err(ImportError::UnsupportedShape { provider });
    }

    Ok(SessionReplay {
        source: replay_source(input, fallback_source),
        protocol,
        turns,
    })
}

/// Build the fixture source text.
fn replay_source(input: &SessionInput<'_>, fallback: &str) -> String {
    input.source_name().map_or_else(
        || fallback.to_owned(),
        |source_name| format!("{fallback}: {source_name}"),
    )
}

impl SessionReplay {
    /// Load a session replay fixture relative to
    /// `tests/integration/fixtures/`.
    ///
    /// # Panics
    ///
    /// Panics if the file cannot be read or parsed, or if the replay has
    /// no turns.
    pub fn load(relative_path: &str) -> Self {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("..");
        path.push("integration");
        path.push("fixtures");
        for component in std::path::Path::new(relative_path).components() {
            let Component::Normal(component) = component else {
                panic!("invalid session replay fixture path {relative_path:?}");
            };
            path.push(component);
        }

        let content = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read fixture {}: {e}", path.display()));
        let replay: Self =
            serde_json::from_str(&content).unwrap_or_else(|e| panic!("parse fixture {relative_path}: {e}"));
        assert!(
            !replay.turns.is_empty(),
            "session replay fixture {relative_path} must have at least one turn"
        );
        replay
    }

    /// Return the only turn in a single-turn replay fixture.
    ///
    /// # Panics
    ///
    /// Panics if the replay fixture has zero or multiple turns.
    pub fn single_turn(&self) -> &ReplayTurn {
        assert_eq!(
            self.turns.len(),
            1,
            "single-turn replay fixture should contain exactly one turn"
        );
        &self.turns[0]
    }
}

impl ReplayTurn {
    /// Return the HTTP path for this replay turn.
    pub fn path(&self) -> &str {
        &self.path
    }

    /// Return the request body as compact JSON.
    ///
    /// # Panics
    ///
    /// Panics if the request value cannot be serialized.
    pub fn request_body(&self) -> String {
        serde_json::to_string(&self.request).unwrap_or_else(|e| panic!("serialize replay request: {e}"))
    }

    /// Return the response body as compact JSON.
    ///
    /// # Panics
    ///
    /// Panics if the response value cannot be serialized.
    pub fn response_body(&self) -> String {
        serde_json::to_string(&self.response).unwrap_or_else(|e| panic!("serialize replay response: {e}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CODEX_JSONL: &str = r#"{"timestamp":"2026-07-07T00:00:00Z","type":"session_meta","payload":{"id":"session_import_codex"}}
{"timestamp":"2026-07-07T00:00:01Z","type":"response_item","payload":{"request":{"model":"gpt-4.1","input":"hello"},"response":{"id":"resp_import_codex","object":"response","status":"completed","model":"gpt-4.1","output":[{"type":"message","content":[{"type":"output_text","text":"hi"}]}]}}}"#;

    const CLAUDE_CODE_JSONL: &str = r#"{"uuid":"turn_user","sessionId":"session_import_claude","timestamp":"2026-07-07T00:00:00Z","type":"user","message":{"role":"user","content":"hello"}}
{"uuid":"turn_assistant","sessionId":"session_import_claude","timestamp":"2026-07-07T00:00:01Z","type":"assistant","message":{"id":"msg_import_claude","type":"message","role":"assistant","model":"claude-sonnet-4-5","content":[{"type":"text","text":"hi"}],"stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":1,"output_tokens":1}}}"#;

    #[test]
    fn load_claude_messages_replay() {
        let replay = SessionReplay::load("replay/claude/messages-basic.json");

        assert_eq!(replay.protocol, ReplayProtocol::AnthropicMessages);
        assert_eq!(replay.turns.len(), 1);
        assert_eq!(replay.single_turn().path(), "/v1/messages");
    }

    #[test]
    fn load_codex_responses_replay() {
        let replay = SessionReplay::load("replay/codex/responses-basic.json");

        assert_eq!(replay.protocol, ReplayProtocol::OpenaiResponses);
        assert_eq!(replay.turns.len(), 1);
        assert_eq!(replay.single_turn().path(), "/v1/responses");
    }

    #[test]
    #[should_panic(expected = "invalid session replay fixture path")]
    fn load_rejects_parent_directory_components() {
        let _replay = SessionReplay::load("../replay/codex/responses-basic.json");
    }

    #[test]
    fn bodies_are_valid_json() {
        let replay = SessionReplay::load("replay/codex/responses-basic.json");
        let turn = replay.single_turn();

        serde_json::from_str::<Value>(&turn.request_body()).expect("request body should be JSON");
        serde_json::from_str::<Value>(&turn.response_body()).expect("response body should be JSON");
    }

    #[test]
    fn session_replay_import_detects_codex_jsonl() {
        let input = SessionInput::new(CODEX_JSONL);

        let detection = CodexSessionImporter.detect(&input);

        assert_eq!(detection, Detection::Detected(SessionProvider::Codex));
    }

    #[test]
    fn session_replay_import_detects_claude_code_jsonl() {
        let input = SessionInput::new(CLAUDE_CODE_JSONL);

        let detection = ClaudeCodeSessionImporter.detect(&input);

        assert_eq!(detection, Detection::Detected(SessionProvider::ClaudeCode));
    }

    #[test]
    fn session_replay_import_converts_codex_turns() {
        let input = SessionInput::new(CODEX_JSONL);

        let replay = import_session_replay(&input, ImportOptions::default()).expect("import should succeed");
        let turn = replay.single_turn();

        assert_eq!(replay.protocol, ReplayProtocol::OpenaiResponses);
        assert_eq!(turn.path(), "/v1/responses");
        assert_eq!(turn.request["input"], "hello");
        assert_eq!(turn.response["id"], "resp_import_codex");
    }

    #[test]
    fn session_replay_import_converts_claude_code_turns() {
        let input = SessionInput::new(CLAUDE_CODE_JSONL);

        let replay = import_session_replay(&input, ImportOptions::default()).expect("import should succeed");
        let turn = replay.single_turn();

        assert_eq!(replay.protocol, ReplayProtocol::AnthropicMessages);
        assert_eq!(turn.path(), "/v1/messages");
        assert_eq!(turn.request["model"], "claude-sonnet-4-5");
        assert_eq!(turn.request["messages"][0]["role"], "user");
        assert_eq!(turn.response["id"], "msg_import_claude");
    }
}
