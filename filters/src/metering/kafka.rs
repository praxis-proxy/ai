// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Kafka usage-event sink backed by `rdkafka` (feature `metering-kafka`).

use std::time::Duration;

use metrics::counter;
use praxis_ai_apis::credential_source::{self, CredentialSource, DiagnosticContext};
use praxis_filter::FilterError;
use rdkafka::{
    ClientConfig,
    producer::{FutureProducer, FutureRecord, Producer},
    util::Timeout,
};
use tracing::{trace, warn};
use zeroize::Zeroizing;

use super::{
    METRIC_REPORT_FAILURES,
    config::{KafkaConfig, KafkaSaslMechanism, join_kafka_brokers, kafka_security_protocol},
};

/// Bounded wait when flushing the producer during filter teardown.
const KAFKA_FLUSH_TIMEOUT: Duration = Duration::from_secs(5);

/// Fire-and-forget Kafka producer for metering `CloudEvents`.
pub(super) struct KafkaUsageEventSink {
    /// Idempotent `rdkafka` producer shared across fire-and-forget tasks.
    producer: Option<FutureProducer>,
    /// Topic that receives `CloudEvent` JSON payloads.
    topic: String,
}

impl KafkaUsageEventSink {
    /// Build an idempotent producer from validated Kafka config.
    ///
    /// # Errors
    /// Returns [`FilterError`] when the SASL password cannot be read or the
    /// producer fails to be constructed.
    pub(super) fn new(cfg: &KafkaConfig, timeout: Duration) -> Result<Self, FilterError> {
        let password = read_sasl_password(&cfg.sasl.password)?;
        let brokers = join_kafka_brokers(&cfg.brokers);
        let security_protocol = kafka_security_protocol(cfg.tls.enabled);
        let message_timeout_ms = u64::try_from(timeout.as_millis().max(1))
            .unwrap_or(u64::MAX)
            .to_string();

        let mechanism = match cfg.sasl.mechanism {
            KafkaSaslMechanism::ScramSha512 => "SCRAM-SHA-512",
        };

        let mut client = ClientConfig::new();
        client
            .set("bootstrap.servers", &brokers)
            .set("acks", "all")
            .set("enable.idempotence", "true")
            .set("message.timeout.ms", &message_timeout_ms)
            .set("security.protocol", security_protocol)
            .set("sasl.mechanisms", mechanism)
            .set("sasl.username", &cfg.sasl.username)
            .set("sasl.password", password.as_str());

        if cfg.tls.enabled {
            let ca_file = cfg
                .tls
                .ca_file
                .as_deref()
                .ok_or_else(|| FilterError::from("external_metering: kafka.tls.ca_file is required"))?;
            client.set("ssl.ca.location", ca_file);
        }

        let producer: FutureProducer = client.create().map_err(|error| {
            FilterError::from(format!("external_metering: failed to create Kafka producer: {error}"))
        })?;

        Ok(Self {
            producer: Some(producer),
            topic: cfg.topic.clone(),
        })
    }

    /// Enqueue event immediately, then spawn a task to await delivery.
    pub(super) fn deliver(&self, body: Vec<u8>, event_id: String) {
        let record = FutureRecord::to(&self.topic).payload(&body).key(&event_id);
        let Some(producer) = self.producer.as_ref() else {
            return;
        };
        let delivery = match producer.send_result(record) {
            Ok(delivery) => delivery,
            Err((error, _)) => {
                counter!(METRIC_REPORT_FAILURES).increment(1);
                warn!(%error, event_id = %event_id, "metering Kafka usage report enqueue failed");
                return;
            },
        };

        tokio::spawn(async move {
            match delivery.await {
                Ok(Ok(delivery)) => {
                    trace!(
                        partition = delivery.partition,
                        offset = delivery.offset,
                        event_id = %event_id,
                        "metering usage report produced to Kafka"
                    );
                },
                Ok(Err((error, _))) => {
                    counter!(METRIC_REPORT_FAILURES).increment(1);
                    warn!(%error, event_id = %event_id, "metering Kafka usage report failed");
                },
                Err(_) => {
                    counter!(METRIC_REPORT_FAILURES).increment(1);
                    warn!(
                        event_id = %event_id,
                        "metering Kafka usage report canceled before delivery status"
                    );
                },
            }
        });
    }
}

impl Drop for KafkaUsageEventSink {
    fn drop(&mut self) {
        let Some(producer) = self.producer.take() else {
            return;
        };
        let teardown = move || {
            if let Err(error) = producer.flush(Timeout::After(KAFKA_FLUSH_TIMEOUT)) {
                warn!(%error, "metering Kafka producer flush on teardown failed");
            }
            // `FutureProducer` owns a polling thread and joins it on drop, so
            // destroy it on the same blocking worker as the flush.
            drop(producer);
        };

        // Do not queue this on Tokio's blocking pool: the runtime may already
        // be shutting down, in which case queued blocking work is never run.
        // Moving ownership also keeps the final producer's polling-thread join
        // off the service worker that drops this sink.
        drop(
            std::thread::Builder::new()
                .name("metering-kafka-teardown".to_owned())
                .spawn(teardown),
        );
    }
}

/// Read the SASL password from a file-backed credential source.
fn read_sasl_password(source: &CredentialSource) -> Result<Zeroizing<String>, FilterError> {
    let ctx = DiagnosticContext {
        owner: "external_metering",
        reference: Some("kafka.sasl.password".to_owned()),
    };
    let transform = |token: &str| -> Result<Zeroizing<String>, FilterError> { Ok(Zeroizing::new(token.to_owned())) };
    let (handle, _watch) = credential_source::build_one(ctx, source, transform)?;
    let snapshot = handle.current();
    snapshot
        .payload
        .as_ref()
        .cloned()
        .ok_or_else(|| FilterError::from("external_metering: kafka.sasl.password is unavailable after startup read"))
}
