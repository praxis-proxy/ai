// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Redpanda-backed Kafka lifecycle for integration tests.
//!
//! Starts a single-broker container with SCRAM-SHA-512 enabled and removes it
//! on drop. Requires `docker` or `podman` on `$PATH` (or `CONTAINER_ENGINE`).

use std::{
    fs,
    process::Command,
    thread,
    time::{Duration, Instant},
};

use rdkafka::{
    ClientConfig,
    consumer::{BaseConsumer, Consumer},
    message::Message,
};
use tempfile::TempDir;

use super::free_port;

const REDPANDA_IMAGE: &str = "docker.io/redpandadata/redpanda:v24.3.9";
const USERNAME: &str = "metering-producer";
const PASSWORD: &str = "test-password";
const READY_TIMEOUT: Duration = Duration::from_secs(90);
const POLL_INTERVAL: Duration = Duration::from_millis(100);

/// RAII guard for a single-broker Redpanda container.
pub struct KafkaGuard {
    /// Temporary bootstrap configuration mounted while the broker starts.
    _bootstrap: TempDir,
    /// Container engine command (`docker` or `podman`).
    engine: String,
    /// Container ID returned by the engine.
    container_id: String,
    /// Host port mapped to the broker's external Kafka listener.
    port: u16,
}

impl KafkaGuard {
    /// Kafka bootstrap address reachable from the test process.
    pub fn bootstrap_servers(&self) -> String {
        format!("127.0.0.1:{}", self.port)
    }

    /// Consume exactly `count` topic records before `timeout` expires.
    pub fn consume_messages(&self, topic: &str, count: usize, timeout: Duration) -> Vec<Vec<u8>> {
        let bootstrap_servers = self.bootstrap_servers();
        let group_id = format!("praxis-metering-test-{}", std::process::id());
        let consumer: BaseConsumer = ClientConfig::new()
            .set("bootstrap.servers", &bootstrap_servers)
            .set("group.id", &group_id)
            .set("enable.auto.commit", "false")
            .set("auto.offset.reset", "earliest")
            .set("security.protocol", "SASL_PLAINTEXT")
            .set("sasl.mechanisms", "SCRAM-SHA-512")
            .set("sasl.username", USERNAME)
            .set("sasl.password", PASSWORD)
            .create()
            .expect("create Kafka test consumer");
        consumer.subscribe(&[topic]).expect("subscribe Kafka test consumer");

        let deadline = Instant::now() + timeout;
        let mut messages = Vec::with_capacity(count);
        while messages.len() < count {
            let remaining = deadline.saturating_duration_since(Instant::now());
            assert!(
                !remaining.is_zero(),
                "timed out after receiving {} of {count} Kafka records from {topic}",
                messages.len()
            );

            if let Some(result) = consumer.poll(remaining.min(POLL_INTERVAL)) {
                let message = result.expect("Kafka consumer poll");
                messages.push(message.payload().expect("metering Kafka record payload").to_vec());
            }
        }

        messages
    }
}

impl Drop for KafkaGuard {
    fn drop(&mut self) {
        let _ = Command::new(&self.engine).args(["kill", &self.container_id]).output();
    }
}

/// Start a Kafka-compatible Redpanda broker with SCRAM-SHA-512 enabled.
pub fn start_kafka() -> KafkaGuard {
    let engine = super::postgres::detect_container_engine();
    let port = free_port();
    let bootstrap = tempfile::tempdir().expect("create Kafka bootstrap config directory");
    let bootstrap_path = bootstrap.path().join("bootstrap.yaml");
    fs::write(
        &bootstrap_path,
        "enable_sasl: true\nsuperusers:\n  - metering-producer\nauto_create_topics_enabled: true\n",
    )
    .expect("write Kafka bootstrap config");

    let publish = format!("127.0.0.1:{port}:19092");
    let mount = format!("{}:/etc/redpanda/.bootstrap.yaml:ro", bootstrap_path.display());
    let advertise = format!("internal://127.0.0.1:9092,external://127.0.0.1:{port}");
    let output = Command::new(&engine)
        .args(["run", "--detach", "--rm", "--publish", &publish])
        .args([
            "--env",
            "RP_BOOTSTRAP_USER=metering-producer:test-password:SCRAM-SHA-512",
        ])
        .args(["--volume", &mount, REDPANDA_IMAGE, "redpanda", "start"])
        .args(["--mode", "dev-container", "--smp", "1", "--memory", "1G"])
        .args([
            "--kafka-addr",
            "internal://0.0.0.0:9092,external://0.0.0.0:19092",
            "--advertise-kafka-addr",
            &advertise,
        ])
        .output()
        .expect("failed to execute container engine for Kafka");
    assert!(
        output.status.success(),
        "container engine failed to start Kafka: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let container_id = String::from_utf8(output.stdout)
        .expect("Kafka container ID should be valid UTF-8")
        .trim()
        .to_owned();
    let guard = KafkaGuard {
        _bootstrap: bootstrap,
        engine,
        container_id,
        port,
    };
    wait_for_kafka(&guard);
    guard
}

/// Wait until the authenticated Kafka API responds to `rpk` inside the broker.
fn wait_for_kafka(kafka: &KafkaGuard) {
    let deadline = Instant::now() + READY_TIMEOUT;
    let mut last_error = String::new();
    while Instant::now() < deadline {
        let output = Command::new(&kafka.engine)
            .args([
                "exec",
                &kafka.container_id,
                "rpk",
                "-X",
                "brokers=127.0.0.1:9092",
                "-X",
                "user=metering-producer",
                "-X",
                "pass=test-password",
                "-X",
                "sasl.mechanism=SCRAM-SHA-512",
                "cluster",
                "info",
            ])
            .output();
        match output {
            Ok(output) if output.status.success() => return,
            Ok(output) => last_error = String::from_utf8_lossy(&output.stderr).into_owned(),
            Err(error) => last_error = error.to_string(),
        }
        thread::sleep(POLL_INTERVAL);
    }
    panic!("Kafka test broker did not become ready: {last_error}");
}
