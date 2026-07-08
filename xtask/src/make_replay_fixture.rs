// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! `cargo xtask make-replay-fixture` converts stored agent sessions into
//! replay fixtures.

use std::path::{Path, PathBuf};

use clap::Parser;
use praxis_test_utils::{ImportOptions, ProviderHint, SessionInput, import_session_replay};

// -----------------------------------------------------------------------------
// CLI Arguments
// -----------------------------------------------------------------------------

/// CLI arguments for `cargo xtask make-replay-fixture`.
#[derive(Parser)]
pub(crate) struct Args {
    /// Claude Code or Codex JSONL session log to import.
    input: PathBuf,

    /// Provider hint: `auto`, `claude`, `claude_code`, or `codex`.
    #[arg(long, default_value = "auto")]
    provider: ProviderHint,

    /// Output fixture path. Prints to stdout when omitted.
    #[arg(long)]
    out: Option<PathBuf>,
}

// -----------------------------------------------------------------------------
// Entry Point
// -----------------------------------------------------------------------------

/// Convert a session log into a replay fixture.
pub(crate) fn run(args: Args) {
    if let Err(err) = run_inner(args) {
        eprintln!("{err}");
        std::process::exit(1);
    }
}

/// Fallible implementation for [`run`].
fn run_inner(args: Args) -> Result<(), String> {
    let Args { input, provider, out } = args;
    let content = std::fs::read_to_string(&input).map_err(|err| format!("read {}: {err}", input.display()))?;
    let source_name = fixture_source_name(&input);
    let output = render_fixture(&content, &source_name, provider)?;

    if let Some(path) = out {
        write_fixture(&path, &output).map_err(|err| format!("write {}: {err}", path.display()))?;
    } else {
        print!("{output}");
    }

    Ok(())
}

// -----------------------------------------------------------------------------
// Fixture Rendering
// -----------------------------------------------------------------------------

/// Render replay fixture JSON from raw session content.
fn render_fixture(content: &str, source_name: &str, provider: ProviderHint) -> Result<String, String> {
    let options = ImportOptions {
        provider,
        ..ImportOptions::default()
    };
    let input = SessionInput::new(content).with_source_name(source_name);
    let replay = import_session_replay(&input, options).map_err(|err| err.to_string())?;
    let json = serde_json::to_string_pretty(&replay).map_err(|err| format!("serialize replay fixture: {err}"))?;
    Ok(format!("{json}\n"))
}

/// Return a commit-safe source label for generated fixtures.
fn fixture_source_name(input: &Path) -> String {
    input
        .file_name()
        .map_or_else(|| "session-log".to_owned(), |name| name.to_string_lossy().into_owned())
}

/// Write fixture content to `path`, creating parent directories as needed.
fn write_fixture(path: &Path, content: &str) -> std::io::Result<()> {
    if let Some(parent) = path.parent().filter(|parent| !parent.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, content)
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::expect_used, clippy::unwrap_used, reason = "tests")]
mod tests {
    use praxis_test_utils::{ProviderHint, ReplayProtocol, SessionReplay};

    use super::*;

    const CODEX_JSONL: &str = r#"{"timestamp":"2026-07-07T00:00:00Z","type":"session_meta","payload":{"id":"session_import_codex"}}
{"timestamp":"2026-07-07T00:00:01Z","type":"response_item","payload":{"request":{"model":"gpt-4.1","input":"hello"},"response":{"id":"resp_import_codex","object":"response","status":"completed","model":"gpt-4.1","output":[]}}}"#;

    #[test]
    fn make_replay_fixture_renders_pretty_json() {
        let output = render_fixture(CODEX_JSONL, "codex.jsonl", ProviderHint::Codex).expect("render should succeed");

        let replay: SessionReplay = serde_json::from_str(&output).expect("output should be a replay fixture");
        assert_eq!(replay.protocol, ReplayProtocol::OpenaiResponses);
        assert!(output.contains("\n  \"source\""), "fixture should be pretty-printed");
    }

    #[test]
    fn make_replay_fixture_source_name_uses_input_filename() {
        let source = fixture_source_name(Path::new("/Users/example/.codex/sessions/2026/07/07/session.jsonl"));

        assert_eq!(source, "session.jsonl");
    }

    #[cfg(unix)]
    #[test]
    fn make_replay_fixture_source_name_uses_lossy_filename() {
        use std::{ffi::OsStr, os::unix::ffi::OsStrExt as _};

        let source = fixture_source_name(&Path::new("/tmp").join(OsStr::from_bytes(b"session-\xff.jsonl")));

        assert!(source.starts_with("session-"));
        assert!(source.ends_with(".jsonl"));
        assert!(!source.contains("/tmp"));
    }

    #[test]
    fn make_replay_fixture_source_name_uses_neutral_fallback_without_filename() {
        let source = fixture_source_name(Path::new(""));

        assert_eq!(source, "session-log");
    }

    #[test]
    fn make_replay_fixture_writes_parent_directories() {
        let dir = tempfile::tempdir().expect("tempdir should be created");
        let path = dir.path().join("nested").join("fixture.json");

        write_fixture(&path, "{}\n").expect("write should succeed");

        let written = std::fs::read_to_string(path).expect("fixture should be readable");
        assert_eq!(written, "{}\n");
    }
}
