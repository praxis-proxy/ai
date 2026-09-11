// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! ClientTestHarness — unified test harness for pinned CLI clients
//! (Claude Code and Codex CLI) in integration tests.

use std::{
    fs,
    path::Path,
    process::{Command, ExitStatus},
};

use tempfile::TempDir;

/// Pinned version of the Claude Code CLI executable used for E2E acceptance tests.
pub(crate) const CLAUDE_CODE_PINNED_VERSION: &str = "2.1.267";

/// Isolated temporary workspace seeded for deterministically verifying client execution.
pub(crate) struct TempWorkspace {
    dir: TempDir,
    expected_content: String,
}

impl TempWorkspace {
    /// Create a new workspace seeded with `input.json`, empty `result.txt`, and executable `verify.sh`.
    pub(crate) fn new() -> std::io::Result<Self> {
        let dir = TempDir::new()?;
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        let expected_content = format!("SUCCESS_E2E_TEST_{nonce}");

        let input_json = serde_json::json!({
            "version": "3.6.0-mvp",
            "target_file": "result.txt",
            "expected_content": expected_content
        });

        fs::write(
            dir.path().join("input.json"),
            serde_json::to_string_pretty(&input_json)?,
        )?;
        fs::write(dir.path().join("result.txt"), "")?;

        let verify_sh = format!("#!/bin/sh\ngrep -q \"{expected_content}\" result.txt\n");
        let verify_path = dir.path().join("verify.sh");
        fs::write(&verify_path, verify_sh)?;

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt as _;
            let mut perms = fs::metadata(&verify_path)?.permissions();
            perms.set_mode(0o755);
            fs::set_permissions(&verify_path, perms)?;
        }

        run_git_cmd(&["init"], dir.path());
        run_git_cmd(&["config", "user.name", "Test"], dir.path());
        run_git_cmd(&["config", "user.email", "test@example.com"], dir.path());
        run_git_cmd(&["add", "."], dir.path());
        run_git_cmd(&["commit", "-m", "initial"], dir.path());

        Ok(Self { dir, expected_content })
    }

    /// Absolute path to the workspace root.
    pub(crate) fn path(&self) -> &Path {
        self.dir.path()
    }

    /// Expected target content generated for this workspace run.
    pub(crate) fn expected_content(&self) -> &str {
        &self.expected_content
    }

    /// Read the current content of `result.txt`.
    pub(crate) fn read_result(&self) -> std::io::Result<String> {
        fs::read_to_string(self.dir.path().join("result.txt"))
    }

    /// Independently execute `./verify.sh` and return its exit status.
    pub(crate) fn run_verification(&self) -> std::io::Result<ExitStatus> {
        Command::new("./verify.sh").current_dir(self.dir.path()).status()
    }

    /// Assert that `result.txt` contains the target string and `./verify.sh` passes.
    pub(crate) fn assert_successful_completion(&self) {
        let content = self.read_result().expect("failed to read result.txt from workspace");
        assert!(
            content.contains(&self.expected_content),
            "workspace result.txt should contain expected string '{}', got: '{}'",
            self.expected_content,
            content
        );

        let status = self.run_verification().expect("execution of verify.sh script failed");
        assert!(
            status.success(),
            "workspace verify.sh script should exit with status 0, got exit code: {:?}",
            status.code()
        );
    }
}

fn run_git_cmd(args: &[&str], dir: &Path) {
    let status = Command::new("git")
        .args(args)
        .current_dir(dir)
        .status()
        .expect("git command should execute");
    assert!(
        status.success(),
        "git command `git {}` should exit with status 0, got: {:?}",
        args.join(" "),
        status.code()
    );
}
