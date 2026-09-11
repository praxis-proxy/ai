// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Bounded, isolated child-process execution shared by CLI acceptance tests.
//!
//! Real-executable acceptance tests (for example Codex and Claude Code) launch
//! a pinned CLI and must contain it: run it in its own process group so a
//! timeout kills the whole descendant tree, clear its environment before spawn,
//! and cap how many bytes of its standard output and standard error the harness
//! retains. This module owns that control logic so every such test shares one
//! audited implementation instead of copying it.

use std::{process::ExitStatus, time::Duration};

#[cfg(unix)]
use nix::{
    errno::Errno,
    sys::signal::{Signal, kill},
    unistd::Pid,
};
use tokio::{
    io::AsyncReadExt as _,
    process::{Child, Command},
    task::JoinHandle,
};

/// Default per-stream cap on retained child standard output or error bytes.
pub const DEFAULT_MAX_CAPTURED_STREAM_BYTES: usize = 8 * 1024 * 1024;

/// Maximum time allowed for process-group termination and pipe drain.
pub const CHILD_CLEANUP_TIMEOUT: Duration = Duration::from_secs(2);

/// Read buffer size used while draining one child pipe.
const PIPE_READ_CHUNK_BYTES: usize = 8 * 1024;

/// Captured child-process result with timeout and truncation state.
#[derive(Debug)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "each flag reports one independent capture outcome: timeout and per-stream truncation"
)]
pub struct CapturedChildOutput {
    /// Child exit status.
    pub status: ExitStatus,
    /// Retained standard output, capped at the configured stream limit.
    pub stdout: Vec<u8>,
    /// Retained standard error, capped at the configured stream limit.
    pub stderr: Vec<u8>,
    /// Whether the child exceeded its execution timeout and was terminated.
    pub timed_out: bool,
    /// Whether retained standard output was truncated at the stream limit.
    pub stdout_truncated: bool,
    /// Whether retained standard error was truncated at the stream limit.
    pub stderr_truncated: bool,
}

/// One capped pipe drain result.
struct CappedPipe {
    /// Retained bytes up to the configured limit.
    bytes: Vec<u8>,
    /// Whether the stream produced more bytes than the limit retained.
    truncated: bool,
}

/// Put a child in its own process group so timeout cleanup includes descendants.
#[cfg(unix)]
pub fn configure_isolated_process_group(command: &mut Command) {
    use std::os::unix::process::CommandExt as _;

    command.as_std_mut().process_group(0);
}

/// Preserve the direct-child behavior on platforms without process groups.
#[cfg(not(unix))]
pub fn configure_isolated_process_group(_command: &mut Command) {}

/// Wait for a child within a timeout and capture both pipes at the default cap.
///
/// See [`capture_child_output_with_limit`] for the per-stream byte ceiling and
/// timeout-termination semantics.
pub async fn capture_child_output(child: Child, execution_timeout: Duration) -> CapturedChildOutput {
    capture_child_output_with_limit(child, execution_timeout, DEFAULT_MAX_CAPTURED_STREAM_BYTES).await
}

/// Wait for a child, terminate its process group on timeout, and cap both pipes.
///
/// Both pipes are drained to end-of-file even after the retained buffer reaches
/// `max_stream_bytes`, so a chatty child never blocks on a full pipe and can
/// always reach exit; excess bytes are discarded and flagged as truncated.
///
/// # Panics
///
/// Panics if the child's standard output or error was not piped, if the child
/// cannot be waited on or reaped, if a timed-out process group cannot be
/// killed, or if a pipe reader task cannot be joined within the cleanup bound.
pub async fn capture_child_output_with_limit(
    mut child: Child,
    execution_timeout: Duration,
    max_stream_bytes: usize,
) -> CapturedChildOutput {
    let process_group_id = child.id();
    let mut stdout = child.stdout.take().expect("stdout should be piped");
    let mut stderr = child.stderr.take().expect("stderr should be piped");
    let mut stdout_task = tokio::spawn(async move { read_capped(&mut stdout, max_stream_bytes).await });
    let mut stderr_task = tokio::spawn(async move { read_capped(&mut stderr, max_stream_bytes).await });

    let (status, timed_out) = if let Ok(result) = tokio::time::timeout(execution_timeout, child.wait()).await {
        (result.expect("child process should be waitable"), false)
    } else {
        terminate_process_group(process_group_id, &mut child);
        let status = tokio::time::timeout(CHILD_CLEANUP_TIMEOUT, child.wait())
            .await
            .expect("killed child should be reaped within the cleanup timeout")
            .expect("killed child should be waitable");
        (status, true)
    };

    let stdout = collect_pipe(&mut stdout_task, process_group_id, "stdout").await;
    let stderr = collect_pipe(&mut stderr_task, process_group_id, "stderr").await;
    CapturedChildOutput {
        status,
        stdout: stdout.bytes,
        stderr: stderr.bytes,
        timed_out,
        stdout_truncated: stdout.truncated,
        stderr_truncated: stderr.truncated,
    }
}

/// Drains one child pipe to end-of-file while retaining at most `limit` bytes.
async fn read_capped<R>(mut reader: R, limit: usize) -> CappedPipe
where
    R: tokio::io::AsyncRead + Unpin,
{
    let mut bytes = Vec::new();
    let mut truncated = false;
    // Heap-allocate the drain buffer so the spawned reader future stays small.
    let mut buffer = vec![0_u8; PIPE_READ_CHUNK_BYTES];
    loop {
        let read = reader.read(&mut buffer).await.expect("child pipe should be readable");
        if read == 0 {
            break;
        }
        let remaining = limit.saturating_sub(bytes.len());
        if remaining == 0 {
            truncated = true;
            continue;
        }
        let take = remaining.min(read);
        bytes.extend_from_slice(&buffer[..take]);
        if take < read {
            truncated = true;
        }
    }
    CappedPipe { bytes, truncated }
}

/// Terminate an isolated child process group, falling back to the direct child.
fn terminate_process_group(process_group_id: Option<u32>, child: &mut Child) {
    #[cfg(unix)]
    if let Some(id) = process_group_id {
        let id = i32::try_from(id).expect("child PID should fit in i32");
        match kill(Pid::from_raw(-id), Signal::SIGKILL) {
            Ok(()) | Err(Errno::ESRCH) => return,
            Err(error) => panic!("timed-out child process group should be killable: {error}"),
        }
    }

    #[cfg(not(unix))]
    let _ = process_group_id;

    child.start_kill().expect("timed-out child process should be killable");
}

/// Collect one capped pipe within a bound, killing inherited descendants if needed.
async fn collect_pipe(task: &mut JoinHandle<CappedPipe>, process_group_id: Option<u32>, name: &str) -> CappedPipe {
    if let Ok(result) = tokio::time::timeout(CHILD_CLEANUP_TIMEOUT, &mut *task).await {
        return result.unwrap_or_else(|error| panic!("{name} reader should finish: {error}"));
    }

    #[cfg(unix)]
    if let Some(id) = process_group_id {
        let id = i32::try_from(id).expect("child PID should fit in i32");
        match kill(Pid::from_raw(-id), Signal::SIGKILL) {
            Ok(()) | Err(Errno::ESRCH) => {},
            Err(error) => panic!("descendant process group holding {name} should be killable: {error}"),
        }
    }

    #[cfg(not(unix))]
    let _ = process_group_id;

    let result = tokio::time::timeout(CHILD_CLEANUP_TIMEOUT, &mut *task).await;
    let Ok(result) = result else {
        task.abort();
        panic!("{name} reader exceeded the cleanup timeout")
    };
    result.unwrap_or_else(|error| panic!("{name} reader should finish: {error}"))
}

#[cfg(all(test, unix))]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(
    clippy::unwrap_used,
    clippy::tests_outside_test_module,
    reason = "tests; module is cfg(all(test, unix)) which the lint does not treat as a test module"
)]
mod tests {
    use std::{process::Stdio, time::Duration};

    use super::{
        CHILD_CLEANUP_TIMEOUT, capture_child_output, capture_child_output_with_limit, configure_isolated_process_group,
    };

    /// Spawns one isolated `/bin/sh` child with both pipes captured.
    fn spawn_shell(script: &str) -> tokio::process::Child {
        let mut command = tokio::process::Command::new("/bin/sh");
        command
            .arg("-c")
            .arg(script)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        configure_isolated_process_group(&mut command);
        command.spawn().expect("shell fixture should start")
    }

    #[tokio::test]
    async fn capped_capture_truncates_and_reports_overflow() {
        // Emit 1000 bytes but retain only 100; the child still exits cleanly
        // because the reader keeps draining the pipe past the retention cap.
        let child = spawn_shell("i=0; while [ $i -lt 1000 ]; do printf A; i=$((i+1)); done");
        let output = capture_child_output_with_limit(child, Duration::from_secs(10), 100).await;

        assert!(output.status.success(), "shell fixture should exit cleanly");
        assert!(!output.timed_out, "bounded fixture should not time out");
        assert_eq!(output.stdout.len(), 100, "stdout must be retained at the cap");
        assert!(output.stdout_truncated, "over-cap stdout must report truncation");
    }

    #[tokio::test]
    async fn small_output_is_retained_without_truncation() {
        let child = spawn_shell("printf PONG");
        let output = capture_child_output(child, Duration::from_secs(10)).await;

        assert!(output.status.success(), "shell fixture should exit cleanly");
        assert_eq!(output.stdout, b"PONG", "under-cap stdout must be retained verbatim");
        assert!(!output.stdout_truncated, "under-cap stdout must not report truncation");
        assert!(output.stderr.is_empty(), "fixture writes nothing to stderr");
    }

    #[tokio::test]
    async fn timed_out_child_kills_process_group_and_closes_inherited_pipes() {
        // A backgrounded `sleep` inherits the piped stdout; unless the whole
        // process group is killed, the pipe reader would block past cleanup.
        let child = spawn_shell("sleep 30 & wait");
        let output = tokio::time::timeout(
            CHILD_CLEANUP_TIMEOUT + Duration::from_secs(2),
            capture_child_output(child, Duration::from_millis(25)),
        )
        .await
        .expect("process-group cleanup and pipe collection should be bounded");

        assert!(output.timed_out, "shell fixture should hit the execution timeout");
        assert!(!output.status.success(), "terminated shell fixture should fail");
    }
}
