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

/// Isolated temporary workspace seeded for deterministically verifying client execution.
pub(crate) struct TempWorkspace {
    dir: TempDir,
    expected_content: String,
}

impl TempWorkspace {
    /// Create a new workspace seeded with `input.json`, empty `result.txt`, and executable `verify.sh`.
    pub(crate) fn new() -> std::io::Result<Self> {
        let dir = TempDir::new()?;
        let expected_content = "SUCCESS_E2E_TEST".to_owned();

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

        let verify_sh = "#!/bin/sh\ngrep -q \"SUCCESS_E2E_TEST\" result.txt\n";
        let verify_path = dir.path().join("verify.sh");
        fs::write(&verify_path, verify_sh)?;

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt as _;
            let mut perms = fs::metadata(&verify_path)?.permissions();
            perms.set_mode(0o755);
            fs::set_permissions(&verify_path, perms)?;
        }

        Ok(Self { dir, expected_content })
    }

    /// Absolute path to the workspace root.
    pub(crate) fn path(&self) -> &Path {
        self.dir.path()
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
    #[expect(
        dead_code,
        reason = "harness helper method provided for client workspace verification"
    )]
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
