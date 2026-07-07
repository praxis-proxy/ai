// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Load stored-session replay fixtures for integration tests.
//!
//! A replay fixture captures one or more sanitized turns from an agent
//! session and replays those turns through an example configuration.

use serde::Deserialize;

/// Stored-session protocol represented by a replay fixture.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ReplayProtocol {
    /// Anthropic Messages API traffic.
    AnthropicMessages,
    /// OpenAI Responses API traffic.
    OpenaiResponses,
}

/// A stored-session replay fixture loaded from JSON.
#[derive(Debug, Deserialize)]
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
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReplayTurn {
    /// Stable fixture-local turn name.
    pub name: String,
    /// HTTP path to replay this turn against.
    pub path: String,
    /// Request body sent by the agent.
    pub request: serde_json::Value,
    /// JSON response body returned by the upstream model service.
    pub response: serde_json::Value,
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
        let mut path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("..");
        path.push("integration");
        path.push("fixtures");
        for component in relative_path.split('/') {
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
    fn bodies_are_valid_json() {
        let replay = SessionReplay::load("replay/codex/responses-basic.json");
        let turn = replay.single_turn();

        serde_json::from_str::<serde_json::Value>(&turn.request_body()).expect("request body should be JSON");
        serde_json::from_str::<serde_json::Value>(&turn.response_body()).expect("response body should be JSON");
    }
}
