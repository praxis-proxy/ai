// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Deserialized YAML configuration types for the external metering filter.

#[cfg(feature = "metering-kafka")]
use std::path::Path;

use praxis_ai_apis::credential_source::CredentialSource;
use praxis_filter::FilterError;
use serde::Deserialize;

/// Default timeout for metering calls and Kafka event delivery (5 seconds).
const DEFAULT_TIMEOUT_SECONDS: u64 = 5;

/// Default entitlement feature key for balance checks.
const DEFAULT_FEATURE_KEY: &str = "inference-tokens";

/// Default `CloudEvents` `source` field.
const DEFAULT_SOURCE: &str = "ai-gateway";

/// Default header prefix for tenant identity headers.
const DEFAULT_IDENTITY_HEADER_PREFIX: &str = "x-tenant-";

/// Default metadata namespace the `identity_header_guard` filter writes
/// captured identity headers under.
const DEFAULT_IDENTITY_METADATA_NAMESPACE: &str = "identity";

/// Kafka brokers accept topic names up to 249 UTF-8 bytes.
#[cfg(feature = "metering-kafka")]
const MAX_KAFKA_TOPIC_NAME_BYTES: usize = 249;

/// SASL mechanisms accepted for the Kafka usage-event transport.
///
/// Kept for YAML deserialization when `metering-kafka` is off so operators
/// get a clear feature-gate error instead of an unknown-field reject.
#[derive(Debug, Clone, Copy, Deserialize)]
pub(super) enum KafkaSaslMechanism {
    /// SCRAM-SHA-512 (the only mechanism accepted by this filter).
    #[serde(rename = "SCRAM-SHA-512")]
    ScramSha512,
}

/// Deserialized YAML config for the `external_metering` filter.
///
/// ```yaml
/// filter: external_metering
/// metering_url: "http://metering-service:8080"
/// allow_private_endpoint: false
/// timeout_seconds: 5
/// feature_key: "inference-tokens"
/// source: "ai-gateway"
/// provider: "openai"
/// fail_open: true
/// identity_header_prefix: "x-tenant-"
/// identity_metadata_namespace: "identity"
/// default_username: "anonymous"
/// default_model: "unknown"
/// ```
///
/// Optional Kafka transport for usage and error `CloudEvents`.
///
/// ```yaml
/// kafka:
///   brokers: ["kafka:9093"]
///   topic: metering.usage.events
///   tls:
///     enabled: true
///     ca_file: /etc/ssl/kafka/ca.pem
///   sasl:
///     mechanism: SCRAM-SHA-512
///     username: metering-producer
///     password:
///       file: /run/secrets/kafka-sasl-password
/// ```
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ExternalMeteringConfig {
    /// Base URL of the external metering service (required).
    pub metering_url: String,

    /// Allow the metering endpoint to resolve to non-public addresses
    /// (loopback, private, link-local). Defaults to `false`, so callouts
    /// are rejected before connecting unless the operator opts in. Cloud
    /// metadata, unspecified, and multicast addresses remain blocked.
    #[serde(default)]
    pub allow_private_endpoint: bool,

    /// Timeout in seconds for metering HTTP calls and Kafka event delivery.
    #[serde(default = "default_timeout_seconds")]
    pub timeout_seconds: u64,

    /// Entitlement feature key used in balance check URL path.
    #[serde(default = "default_feature_key")]
    pub feature_key: String,

    /// `CloudEvents` `source` field value.
    #[serde(default = "default_source")]
    pub source: String,

    /// Static provider name attributed to emitted `CloudEvents`.
    #[serde(default)]
    pub provider: Option<String>,

    /// When `true` (default), requests proceed if the metering service
    /// is unavailable. When `false`, requests are rejected with 503.
    #[serde(default = "default_true")]
    pub fail_open: bool,

    /// Prefix for tenant identity headers to capture and strip.
    /// Expected headers: `{prefix}username`, `{prefix}group`,
    /// `{prefix}subscription`, `{prefix}model`, `{prefix}organization-id`,
    /// `{prefix}cost-center`.
    #[serde(default = "default_identity_header_prefix")]
    pub identity_header_prefix: String,

    /// Metadata namespace the `identity_header_guard` filter writes captured
    /// identity headers under. Must match that filter's
    /// `metadata_namespace` setting when both run in one pipeline.
    #[serde(default = "default_identity_metadata_namespace")]
    pub identity_metadata_namespace: String,

    /// Fallback username when no identity header is present.
    /// If set, requests without `{prefix}username` are still metered
    /// under this name. If unset, metering is skipped entirely.
    #[serde(default)]
    pub default_username: Option<String>,

    /// Fallback model name when no identity model header is present.
    #[serde(default)]
    pub default_model: Option<String>,

    /// Optional internal bearer token for authenticating metering callouts
    /// (the balance check and the usage report). One of `value | env_var |
    /// file`, read through the shared [`CredentialSource`] core: bounded reads,
    /// restart-free file rotation (ideal for a Kubernetes projected Secret),
    /// and fail-closed publication. When omitted, callouts are sent
    /// unauthenticated (unchanged behavior).
    #[serde(default)]
    pub internal_auth_token: Option<CredentialSource>,

    /// Optional Kafka transport for usage and error `CloudEvents`.
    /// When absent, events are sent via POST `{metering_url}/api/v1/events`.
    /// When present, events are produced to Kafka instead.
    /// Balance checks always use `metering_url` via HTTP.
    #[serde(default)]
    pub kafka: Option<KafkaConfig>,
}

/// Kafka producer settings for usage-event delivery.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
#[cfg_attr(
    not(feature = "metering-kafka"),
    expect(
        dead_code,
        reason = "deserialized for a clear feature-gate error when metering-kafka is off"
    )
)]
pub(super) struct KafkaConfig {
    /// Bootstrap broker list (host:port entries).
    pub brokers: Vec<String>,

    /// Topic that receives `CloudEvent` JSON payloads.
    pub topic: String,

    /// TLS configuration.
    pub tls: KafkaTlsConfig,

    /// Required SASL authentication. TLS selects `SASL_SSL` or
    /// `SASL_PLAINTEXT` for the connection.
    pub sasl: KafkaSaslConfig,
}

/// TLS options for the Kafka producer.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
#[cfg_attr(
    not(feature = "metering-kafka"),
    expect(
        dead_code,
        reason = "deserialized for a clear feature-gate error when metering-kafka is off"
    )
)]
pub(super) struct KafkaTlsConfig {
    /// Whether to use TLS for the brokers.
    pub enabled: bool,

    /// PEM CA bundle used for broker certificate verification.
    /// Required when `tls.enabled` is `true` and ignored when TLS is off.
    #[serde(default)]
    pub ca_file: Option<String>,
}

/// SASL options for the Kafka producer.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
#[cfg_attr(
    not(feature = "metering-kafka"),
    expect(
        dead_code,
        reason = "deserialized for a clear feature-gate error when metering-kafka is off"
    )
)]
pub(super) struct KafkaSaslConfig {
    /// SASL mechanism. Must be `SCRAM-SHA-512`.
    pub mechanism: KafkaSaslMechanism,

    /// SASL username.
    pub username: String,

    /// SASL password. Must be file-backed [`CredentialSource`] (`file:` only).
    pub password: CredentialSource,
}

/// Validate config at construction time.
pub(super) fn validate_config(cfg: &ExternalMeteringConfig) -> Result<(), FilterError> {
    if cfg.metering_url.is_empty() {
        return Err("external_metering: metering_url must not be empty".into());
    }

    if cfg.timeout_seconds == 0 {
        return Err("external_metering: timeout_seconds must be greater than 0".into());
    }

    if cfg.identity_header_prefix.is_empty() {
        return Err("external_metering: identity_header_prefix must not be empty".into());
    }

    // A prefix with characters that cannot appear in an HTTP header name
    // (e.g. a space) would silently match nothing, leaving tenant identity
    // headers unstripped while the filter reports healthy.
    if http::header::HeaderName::from_bytes(cfg.identity_header_prefix.as_bytes()).is_err() {
        return Err(
            "external_metering: identity_header_prefix must contain only valid HTTP header name characters".into(),
        );
    }

    if cfg.identity_metadata_namespace.is_empty() {
        return Err("external_metering: identity_metadata_namespace must not be empty".into());
    }

    validate_optional_kafka(cfg.kafka.as_ref())
}

/// Reject or fully validate an optional `kafka:` block.
fn validate_optional_kafka(kafka: Option<&KafkaConfig>) -> Result<(), FilterError> {
    match kafka {
        None => Ok(()),
        Some(kafka) => {
            #[cfg(not(feature = "metering-kafka"))]
            {
                let _ = kafka;
                Err("external_metering: kafka transport requires rebuilding with `--features metering-kafka`".into())
            }
            #[cfg(feature = "metering-kafka")]
            {
                validate_kafka_config(kafka)
            }
        },
    }
}

/// Validate the optional Kafka config.
#[cfg(feature = "metering-kafka")]
fn validate_kafka_config(kafka: &KafkaConfig) -> Result<(), FilterError> {
    validate_kafka_brokers(&kafka.brokers)?;

    if kafka.topic.trim().is_empty() {
        return Err("external_metering: kafka.topic must not be empty".into());
    }

    if kafka.topic.len() > MAX_KAFKA_TOPIC_NAME_BYTES {
        return Err(
            format!("external_metering: kafka.topic must be at most {MAX_KAFKA_TOPIC_NAME_BYTES} bytes").into(),
        );
    }

    if kafka.topic == "."
        || kafka.topic == ".."
        || !kafka
            .topic
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
    {
        return Err(
            "external_metering: kafka.topic may contain only ASCII letters, numbers, '.', '_' or '-', and must not be '.' or '..'".into(),
        );
    }

    if kafka.tls.enabled {
        let Some(ca_file) = kafka.tls.ca_file.as_deref() else {
            return Err("external_metering: kafka.tls.ca_file is required when tls.enabled is true".into());
        };
        if ca_file.trim().is_empty() {
            return Err("external_metering: kafka.tls.ca_file must not be empty".into());
        }
        if !Path::new(ca_file).is_file() {
            return Err(format!("external_metering: kafka.tls.ca_file '{ca_file}' is not a readable file").into());
        }
    }

    if kafka.sasl.username.trim().is_empty() {
        return Err("external_metering: kafka.sasl.username must not be empty".into());
    }

    match &kafka.sasl.password {
        CredentialSource::File(path) => {
            if path.trim().is_empty() {
                return Err("external_metering: kafka.sasl.password file path must not be empty".into());
            }
            if !Path::new(path).is_file() {
                return Err(
                    format!("external_metering: kafka.sasl.password file '{path}' is not a readable file").into(),
                );
            }
        },
        CredentialSource::Value(_) | CredentialSource::EnvVar(_) => {
            return Err(
                "external_metering: kafka.sasl.password must use file: (inline value and env_var are not allowed)"
                    .into(),
            );
        },
    }

    Ok(())
}

/// Reject an empty broker list or malformed host[:port] entries.
#[cfg(feature = "metering-kafka")]
pub(super) fn validate_kafka_brokers(brokers: &[String]) -> Result<(), FilterError> {
    if brokers.is_empty() {
        return Err("external_metering: kafka.brokers must not be empty".into());
    }
    for broker in brokers {
        let invalid = || {
            FilterError::from(format!(
                "external_metering: kafka.brokers entry '{broker}' must be a host[:port] address with a valid port (bracket IPv6 literals)"
            ))
        };
        if broker.trim().is_empty()
            || broker.trim() != broker
            || broker.contains([' ', '\t', '\r', '\n', ',', '/', '@'])
            || broker.contains("://")
        {
            return Err(invalid());
        }

        let authority = broker.parse::<http::uri::Authority>().map_err(|_| invalid())?;
        if authority.host().is_empty() {
            return Err(invalid());
        }

        // `Authority::port()` returns `None` for malformed ports as well as
        // absent ports, so inspect the configured suffix before accepting it.
        let port = if broker.starts_with('[') {
            let Some(host_end) = broker.find(']') else {
                return Err(invalid());
            };
            let suffix = &broker[host_end + 1..];
            if suffix.is_empty() {
                None
            } else {
                Some(suffix.strip_prefix(':').ok_or_else(|| invalid())?)
            }
        } else if let Some((host, port)) = broker.rsplit_once(':') {
            if host.contains(':') {
                return Err(invalid());
            }
            Some(port)
        } else {
            None
        };
        if let Some(port) = port {
            let Ok(port) = port.parse::<u16>() else {
                return Err(invalid());
            };
            if port == 0 {
                return Err(invalid());
            }
        }
    }

    Ok(())
}

/// Map `tls.enabled` to librdkafka `security.protocol` (`SASL_SSL` or
/// `SASL_PLAINTEXT`).
#[cfg(feature = "metering-kafka")]
pub(super) fn kafka_security_protocol(tls_enabled: bool) -> &'static str {
    if tls_enabled { "SASL_SSL" } else { "SASL_PLAINTEXT" }
}

/// Join broker addresses into the librdkafka `bootstrap.servers` list.
#[cfg(feature = "metering-kafka")]
pub(super) fn join_kafka_brokers(brokers: &[String]) -> String {
    brokers.join(",")
}

/// Serde default for `timeout_seconds`.
fn default_timeout_seconds() -> u64 {
    DEFAULT_TIMEOUT_SECONDS
}

/// Serde default for `feature_key`.
fn default_feature_key() -> String {
    DEFAULT_FEATURE_KEY.to_owned()
}

/// Serde default for `source`.
fn default_source() -> String {
    DEFAULT_SOURCE.to_owned()
}

/// Serde default for `fail_open`.
fn default_true() -> bool {
    true
}

/// Serde default for `identity_header_prefix`.
fn default_identity_header_prefix() -> String {
    DEFAULT_IDENTITY_HEADER_PREFIX.to_owned()
}

/// Serde default for `identity_metadata_namespace`.
fn default_identity_metadata_namespace() -> String {
    DEFAULT_IDENTITY_METADATA_NAMESPACE.to_owned()
}
