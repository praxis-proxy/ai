// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! `llm-d-inference-sim` container lifecycle for environment tests.
//!
//! Spawns a simulator container on a random host port and removes
//! it on drop. Requires `docker` or `podman` on `$PATH` (or set
//! `CONTAINER_ENGINE` to override detection).

use std::{
    io::{Read as _, Write as _},
    net::TcpStream,
    process::Command,
    thread,
    time::{Duration, Instant},
};

use super::port::free_port;

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// Container image for the llm-d inference simulator.
const SIM_IMAGE: &str = "ghcr.io/llm-d/llm-d-inference-sim:v0.10.0";

/// Default model name served by the simulator.
const DEFAULT_MODEL: &str = "praxis-env-test-model";

/// Maximum time to wait for the simulator to accept connections.
const READY_TIMEOUT: Duration = Duration::from_secs(60);

/// Interval between readiness polls.
const READY_POLL_INTERVAL: Duration = Duration::from_millis(250);

// -----------------------------------------------------------------------------
// SimulatorGuard
// -----------------------------------------------------------------------------

/// RAII guard that manages an `llm-d-inference-sim` container.
///
/// Spawns the simulator on a random host port, waits for HTTP
/// readiness, and kills the container on drop. The container runs
/// with `--rm` so it is also removed after being killed.
///
/// # Panics
///
/// Panics if no container engine is found, if the container fails
/// to start, or if the simulator does not become ready within the
/// timeout.
pub struct SimulatorGuard {
    /// Container ID (short hash from `docker run`).
    container_id: String,

    /// Host port mapped to the simulator's HTTP port.
    port: u16,

    /// Container engine command (`docker` or `podman`).
    engine: String,

    /// Model name served by this simulator instance.
    model: String,
}

impl SimulatorGuard {
    /// Host port mapped to the simulator's HTTP port.
    pub fn port(&self) -> u16 {
        self.port
    }

    /// The model name served by this simulator instance.
    pub fn model(&self) -> &str {
        &self.model
    }

    /// `host:port` address for use as an `endpoint_selector` destination.
    pub fn endpoint(&self) -> String {
        format!("127.0.0.1:{}", self.port)
    }

    /// URL for the simulator's OpenAI-compatible chat completions API.
    pub fn openai_url(&self) -> String {
        format!("http://127.0.0.1:{}/v1/chat/completions", self.port)
    }

    /// URL for the simulator's health endpoint.
    pub fn health_url(&self) -> String {
        format!("http://127.0.0.1:{}/health", self.port)
    }

    /// URL for the simulator's Prometheus metrics endpoint.
    pub fn metrics_url(&self) -> String {
        format!("http://127.0.0.1:{}/metrics", self.port)
    }
}

impl Drop for SimulatorGuard {
    fn drop(&mut self) {
        let _ = Command::new(&self.engine).args(["kill", &self.container_id]).output();
    }
}

// -----------------------------------------------------------------------------
// Public API
// -----------------------------------------------------------------------------

/// Start an `llm-d-inference-sim` container and wait for it to
/// accept connections.
///
/// Uses `CONTAINER_ENGINE` env var if set, otherwise probes for
/// `podman` then `docker` on `$PATH`.
///
/// # Panics
///
/// Panics if no container engine is available, the container fails
/// to start, or the simulator does not become ready within 60
/// seconds.
pub fn start_simulator() -> SimulatorGuard {
    start_simulator_with_model(DEFAULT_MODEL)
}

/// Start an `llm-d-inference-sim` container serving the given model.
///
/// # Panics
///
/// Panics if no container engine is available, the container fails
/// to start, or the simulator does not become ready within 60
/// seconds.
pub fn start_simulator_with_model(model: &str) -> SimulatorGuard {
    let engine = detect_container_engine();
    let port = free_port();
    let container_id = run_container(&engine, port, model);
    let guard = SimulatorGuard {
        container_id,
        port,
        engine,
        model: model.to_owned(),
    };

    wait_for_simulator(port);

    guard
}

// -----------------------------------------------------------------------------
// Internal Helpers
// -----------------------------------------------------------------------------

/// Spawn a detached simulator container on the given port.
fn run_container(engine: &str, port: u16, model: &str) -> String {
    let output = Command::new(engine)
        .args([
            "run",
            "-d",
            "--rm",
            "-p",
            &format!("{port}:8000"),
            SIM_IMAGE,
            &format!("--model={model}"),
            &format!("--served-model-name={model}"),
            "--port=8000",
        ])
        .output()
        .expect("failed to execute container engine");

    assert!(
        output.status.success(),
        "container engine failed to start simulator: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    String::from_utf8(output.stdout)
        .expect("container ID should be valid UTF-8")
        .trim()
        .to_owned()
}

/// Detect the container engine to use.
fn detect_container_engine() -> String {
    if let Ok(engine) = std::env::var("CONTAINER_ENGINE") {
        return engine;
    }

    for candidate in ["podman", "docker"] {
        if Command::new(candidate)
            .arg("--version")
            .output()
            .is_ok_and(|o| o.status.success())
        {
            return candidate.to_owned();
        }
    }

    panic!("no container engine found - install podman or docker, or set CONTAINER_ENGINE");
}

/// Poll until the simulator's health endpoint returns HTTP 200.
#[expect(
    clippy::disallowed_methods,
    reason = "container readiness polling is synchronous; no async runtime available here"
)]
fn wait_for_simulator(port: u16) {
    let addr = format!("127.0.0.1:{port}");
    let deadline = Instant::now() + READY_TIMEOUT;

    loop {
        if health_is_ready(&addr) {
            return;
        }
        if Instant::now() >= deadline {
            break;
        }
        thread::sleep(READY_POLL_INTERVAL);
    }
    panic!("simulator did not become ready within {READY_TIMEOUT:?}");
}

/// Check whether `/health` returns HTTP 200.
fn health_is_ready(addr: &str) -> bool {
    let Ok(mut stream) = TcpStream::connect(addr) else {
        return false;
    };

    drop(stream.set_read_timeout(Some(Duration::from_secs(2))));
    drop(stream.set_write_timeout(Some(Duration::from_secs(2))));

    let request = b"GET /health HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n";
    if stream.write_all(request).is_err() {
        return false;
    }

    let mut buf = [0_u8; 64];
    stream
        .read(&mut buf)
        .is_ok_and(|n| std::str::from_utf8(&buf[..n]).is_ok_and(|response| response.starts_with("HTTP/1.1 200")))
}
