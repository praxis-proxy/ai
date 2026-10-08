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

/// Portable shell fragment that reads bytes on stdin and prints their lowercase
/// sha256 hex digest (no filename column). GNU coreutils ships `sha256sum`, but
/// macOS (where this suite's precompute and the CLI clients' fixed PATH run) ships
/// only `shasum`, so fall back to it. Keep this in sync across every `verify.sh`
/// the suite writes so the client-side self-check matches the harness precompute.
pub(crate) const SHA256_DIGEST_SH: &str =
    "{ if command -v sha256sum >/dev/null 2>&1; then sha256sum; else shasum -a 256; fi; } | cut -d' ' -f1";

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

        let verify_sh = format!(
            "#!/bin/sh\nset -eu\ngrep -q \"{expected_content}\" result.txt\nprintf 'verified\\n' > .verification-ran\n"
        );
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

        let marker = fs::read_to_string(self.dir.path().join(".verification-ran"))
            .expect("client should execute verify.sh and create its marker");
        assert_eq!(marker, "verified\n", "verification marker should be complete");

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

/// Seed `count` uniquely-filled ballast files (`chapter_01.txt`, ...) of roughly
/// `approx_bytes` each in `dir` and return their names in deterministic order.
fn seed_ballast_files(dir: &Path, count: usize, approx_bytes: usize) -> std::io::Result<Vec<String>> {
    const WORDS: &[&str] = &[
        "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet", "kilo", "lima",
        "mike", "november", "oscar", "papa", "quebec", "romeo", "sierra", "tango", "uniform", "victor", "whiskey",
        "xray", "yankee", "zulu", "summit", "harbor", "meridian", "quartz", "lantern", "cobalt",
    ];
    let mut names = Vec::with_capacity(count);
    for index in 1..=count {
        let name = format!("chapter_{index:02}.txt");
        let mut body = String::with_capacity(approx_bytes + 64);
        body.push_str(&format!("# ballast chapter {index}\n"));
        let mut counter = index;
        while body.len() < approx_bytes {
            body.push_str(WORDS[counter % WORDS.len()]);
            counter += 1;
            if counter % 12 == 0 {
                body.push('\n');
            } else {
                body.push(' ');
            }
        }
        body.push('\n');
        fs::write(dir.join(&name), body)?;
        names.push(name);
    }
    Ok(names)
}

/// Workspace proving the Codex client carries a task across its own compaction
/// and completes it correctly against live vLLM.
///
/// Unlike [`TempWorkspace`], the unique marker lives in `secret.txt`, and
/// `verify.sh` compares a sha256 of the written result against a precomputed
/// `expected.hash` instead of embedding the marker. The marker is high-entropy so
/// it cannot be guessed; the task reads it, grows context past the compaction
/// trigger, then re-reads `secret.txt` after compaction to recover the token and
/// write it into `result.txt`. Real self-compaction is proven independently on the
/// wire (the client's summarization POST), and the result write is proven to
/// follow the compaction boundary; the on-disk source makes the final write
/// reliable regardless of whether the model happened to retain the token verbatim.
pub(crate) struct CodexCompactionWorkspace {
    dir: TempDir,
    marker: String,
}

impl CodexCompactionWorkspace {
    /// Create a workspace whose marker lives in `secret.txt`, with a marker-free
    /// hash-based `verify.sh`.
    ///
    /// The workspace is deliberately NOT a git repository: Codex runs with
    /// `--skip-git-repo-check`, so no repo is required.
    pub(crate) fn new() -> std::io::Result<Self> {
        let dir = TempDir::new()?;
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        // ASCII, whitespace-free so shell `tr -d '[:space:]'` normalization is a no-op.
        let marker = format!("CODEX_COMPACTION_MARKER_{nonce}");

        fs::write(dir.path().join("secret.txt"), &marker)?;
        fs::write(dir.path().join("result.txt"), "")?;

        // Precompute the expected hash with the SAME pipeline verify.sh uses, so
        // the marker never appears inside verify.sh or expected.hash on disk.
        let hash_output = Command::new("sh")
            .arg("-c")
            .arg(format!("tr -d '[:space:]' < secret.txt | {SHA256_DIGEST_SH}"))
            .current_dir(dir.path())
            .output()?;
        assert!(
            hash_output.status.success(),
            "sha256 digest of marker should succeed; stderr: {}",
            String::from_utf8_lossy(&hash_output.stderr)
        );
        let hash = String::from_utf8_lossy(&hash_output.stdout).trim().to_owned();
        assert!(!hash.is_empty(), "sha256 digest should be non-empty");
        fs::write(dir.path().join("expected.hash"), format!("{hash}\n"))?;

        let verify_sh = format!(
            "#!/bin/sh\nset -eu\nactual=$(tr -d '[:space:]' < result.txt | {SHA256_DIGEST_SH})\nexpected=$(tr -d '[:space:]' < expected.hash)\ntest \"$actual\" = \"$expected\"\nprintf 'verified\\n' > .verification-ran\n"
        );
        let verify_path = dir.path().join("verify.sh");
        fs::write(&verify_path, verify_sh)?;

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt as _;
            let mut perms = fs::metadata(&verify_path)?.permissions();
            perms.set_mode(0o755);
            fs::set_permissions(&verify_path, perms)?;
        }

        Ok(Self { dir, marker })
    }

    /// Absolute path to the workspace root.
    pub(crate) fn path(&self) -> &Path {
        self.dir.path()
    }

    /// Seed ballast chapters for context growth (see [`seed_ballast_files`]).
    pub(crate) fn seed_context_ballast(&self, count: usize, approx_bytes: usize) -> std::io::Result<Vec<String>> {
        seed_ballast_files(self.dir.path(), count, approx_bytes)
    }

    /// Assert the task reproduced the marker into `result.txt` and ran the verifier.
    ///
    /// The authoritative oracle is the harness-retained marker compared against
    /// `result.txt` in-process — NOT the in-workspace `verify.sh`/`expected.hash`,
    /// which the client could overwrite. `verify.sh` is kept only so the client
    /// has a self-check step; its `.verification-ran` marker merely corroborates
    /// that the client ran it. That the result write follows the client's own
    /// compaction is proven separately on the wire by the transport observer.
    pub(crate) fn assert_successful_completion(&self) {
        let content =
            fs::read_to_string(self.dir.path().join("result.txt")).expect("failed to read result.txt from workspace");
        let normalized: String = content.chars().filter(|character| !character.is_whitespace()).collect();
        assert_eq!(
            normalized, self.marker,
            "result.txt must contain exactly the marker (independent of the in-workspace verify.sh); got: {content:?}"
        );

        let ran = fs::read_to_string(self.dir.path().join(".verification-ran"))
            .expect("client should execute verify.sh and create its marker");
        assert_eq!(ran, "verified\n", "verification marker should be complete");
    }
}
