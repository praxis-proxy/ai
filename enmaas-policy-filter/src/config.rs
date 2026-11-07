#![allow(
    clippy::missing_docs_in_private_items,
    clippy::struct_excessive_bools,
    clippy::too_many_lines,
    reason = "private configuration internals"
)]

use std::{fs, time::Duration};

use http::{HeaderName, HeaderValue};
use praxis_filter::FilterError;
use serde::Deserialize;

const DEFAULT_TIMEOUT_SECONDS: u64 = 5;
const DEFAULT_DECISION_CACHE_TTL_SECONDS: u64 = 5;
const DEFAULT_DECISION_CACHE_MAX_ENTRIES: usize = 10_000;
const DEFAULT_USERNAME_METADATA: &str = "x-tenant-username";
const DEFAULT_GROUP_METADATA: &str = "x-tenant-group";
const DEFAULT_SUBSCRIPTION_METADATA: &str = "x-tenant-subscription";
const DEFAULT_MODEL_METADATA: &str = "model_to_provider.client_model";
const DEFAULT_IDENTITY_MODEL_METADATA: &str = "x-tenant-model";
const DEFAULT_PROMOTED_MODEL_HEADER: &str = "x-model";
const DEFAULT_CLASS_METADATA: &str = "qos.class";
const DEFAULT_SUBJECT_METADATA: &str = "qos.subject";
const DEFAULT_RETRY_AFTER_METADATA: &str = "qos.retry_after";
const DEFAULT_KEY_ALIAS_METADATA: &str = "qos.key_alias";
const DEFAULT_CLASS: &str = "main";
const DEFAULT_METERING_URL: &str = "http://metering-service:8080";
const DEFAULT_SOURCE: &str = "ai-gateway";
const DEFAULT_MODEL: &str = "unknown";
const DEFAULT_RESPONSE_CLASS_HEADER: &str = "x-enmaas-qos-class";
const DEFAULT_CREDENTIAL_HEADER: &str = "authorization";
const DEFAULT_HEADER_PREFIX: &str = "Bearer ";
const DEFAULT_KEY_ENV: &str = "CB_LITELLM_API_KEY";

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum QosEnforcement {
    #[default]
    Soft,
    Off,
    Hard,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum CredentialMode {
    #[default]
    Off,
    Shadow,
    Active,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct EnmaasPolicyConfig {
    #[serde(default)]
    pub allow_private_endpoint: bool,

    #[serde(default = "default_timeout_seconds")]
    pub timeout_seconds: u64,

    #[serde(default = "default_true")]
    pub fail_open: bool,

    #[serde(default = "default_true")]
    pub enabled: bool,

    #[serde(default = "default_decision_cache_ttl_seconds")]
    pub decision_cache_ttl_seconds: u64,

    #[serde(default = "default_decision_cache_max_entries")]
    pub decision_cache_max_entries: usize,

    #[serde(default = "default_username_metadata")]
    pub username_metadata: String,

    #[serde(default = "default_group_metadata")]
    pub group_metadata: String,

    #[serde(default = "default_subscription_metadata")]
    pub subscription_metadata: String,

    #[serde(default)]
    pub default_username: Option<String>,

    #[serde(default = "default_model_metadata")]
    pub model_metadata: String,

    #[serde(default = "default_identity_model_metadata")]
    pub identity_model_metadata: String,

    #[serde(default = "default_promoted_model_header")]
    pub promoted_model_header: String,

    #[serde(default = "default_class_metadata")]
    pub class_metadata: String,

    #[serde(default = "default_subject_metadata")]
    pub subject_metadata: String,

    #[serde(default = "default_retry_after_metadata")]
    pub retry_after_metadata: String,

    #[serde(default = "default_key_alias_metadata")]
    pub key_alias_metadata: String,

    #[serde(default = "default_class")]
    pub default_class: String,

    #[serde(default)]
    pub qos_enforcement: QosEnforcement,

    #[serde(default = "default_allow_missing_model")]
    pub allow_missing_model: bool,

    #[serde(default = "default_metering_url")]
    pub metering_url: String,

    #[serde(default = "default_feature_key")]
    pub metering_feature_key: String,

    #[serde(default)]
    pub metering_auth_file: Option<String>,

    #[serde(default = "default_key_alias_premium")]
    pub key_alias_premium: String,

    #[serde(default = "default_key_alias_main")]
    pub key_alias_main: String,

    #[serde(default = "default_key_alias_ratelimit")]
    pub key_alias_ratelimit: String,

    #[serde(default = "default_source")]
    pub source: String,

    #[serde(default)]
    pub provider: Option<String>,

    #[serde(default = "default_model")]
    pub default_model: String,

    #[serde(default = "default_response_class_header")]
    pub response_class_header: String,

    #[serde(default = "default_main_burst")]
    pub main_burst: u32,

    #[serde(default = "default_main_refill")]
    pub main_refill_per_second: f64,

    #[serde(default)]
    pub premium_burst: u32,

    #[serde(default)]
    pub premium_refill_per_second: f64,

    #[serde(default = "default_max_keys")]
    pub max_keys: usize,

    #[serde(default)]
    pub credential_mode: CredentialMode,

    #[serde(default)]
    pub key_file: Option<String>,

    #[serde(default = "default_key_clusters")]
    pub key_clusters: Vec<String>,

    #[serde(default = "default_qos_clusters")]
    pub qos_clusters: Vec<String>,

    #[serde(default = "default_credential_header")]
    pub credential_header: String,

    #[serde(default = "default_header_prefix")]
    pub header_prefix: String,

    #[serde(default = "default_key_env")]
    pub default_key_env: String,
}

impl EnmaasPolicyConfig {
    pub(crate) fn timeout(&self) -> Duration {
        Duration::from_secs(self.timeout_seconds)
    }

    pub(crate) fn promoted_model_header_name(&self) -> Result<HeaderName, FilterError> {
        HeaderName::from_bytes(self.promoted_model_header.as_bytes()).map_err(|error| -> FilterError {
            format!("enmaas_policy: promoted_model_header is invalid: {error}").into()
        })
    }

    pub(crate) fn metering_auth_token(&self) -> Result<Option<HeaderValue>, FilterError> {
        read_bearer_token(self.metering_auth_file.as_deref(), "metering_auth_file")
    }

    pub(crate) fn response_class_header_name(&self) -> Result<Option<HeaderName>, FilterError> {
        if self.response_class_header.trim().is_empty() {
            return Ok(None);
        }
        HeaderName::from_bytes(self.response_class_header.as_bytes())
            .map(Some)
            .map_err(|error| -> FilterError {
                format!("enmaas_policy: response_class_header is invalid: {error}").into()
            })
    }

    pub(crate) fn credential_header_name(&self) -> Result<HeaderName, FilterError> {
        HeaderName::from_bytes(self.credential_header.as_bytes())
            .map_err(|error| -> FilterError { format!("enmaas_policy: credential_header is invalid: {error}").into() })
    }
}

pub(crate) fn validate_config(cfg: &EnmaasPolicyConfig) -> Result<(), FilterError> {
    if cfg.metering_url.trim().is_empty() {
        return Err("enmaas_policy: metering_url must not be empty".into());
    }
    if cfg.timeout_seconds == 0 {
        return Err("enmaas_policy: timeout_seconds must be greater than 0".into());
    }
    if cfg.decision_cache_ttl_seconds > 0 && cfg.decision_cache_max_entries == 0 {
        return Err("enmaas_policy: decision_cache_max_entries must be greater than 0".into());
    }
    for (name, value) in [
        ("username_metadata", &cfg.username_metadata),
        ("group_metadata", &cfg.group_metadata),
        ("subscription_metadata", &cfg.subscription_metadata),
        ("model_metadata", &cfg.model_metadata),
        ("identity_model_metadata", &cfg.identity_model_metadata),
        ("class_metadata", &cfg.class_metadata),
        ("subject_metadata", &cfg.subject_metadata),
        ("retry_after_metadata", &cfg.retry_after_metadata),
        ("key_alias_metadata", &cfg.key_alias_metadata),
        ("source", &cfg.source),
        ("default_model", &cfg.default_model),
        ("default_key_env", &cfg.default_key_env),
        ("metering_feature_key", &cfg.metering_feature_key),
        ("key_alias_premium", &cfg.key_alias_premium),
        ("key_alias_main", &cfg.key_alias_main),
        ("key_alias_ratelimit", &cfg.key_alias_ratelimit),
    ] {
        if value.trim().is_empty() {
            return Err(format!("enmaas_policy: {name} must not be empty").into());
        }
    }
    if !matches!(cfg.default_class.as_str(), "premium" | "main" | "ratelimit") {
        return Err("enmaas_policy: default_class must be premium, main, or ratelimit".into());
    }
    if cfg.max_keys == 0 {
        return Err("enmaas_policy: max_keys must be greater than 0".into());
    }
    validate_bucket("main", cfg.main_burst, cfg.main_refill_per_second)?;
    validate_bucket("premium", cfg.premium_burst, cfg.premium_refill_per_second)?;
    if cfg.key_clusters.is_empty() || cfg.key_clusters.iter().any(|value| value.trim().is_empty()) {
        return Err("enmaas_policy: key_clusters must contain at least one non-empty name".into());
    }
    if cfg.qos_clusters.is_empty() || cfg.qos_clusters.iter().any(|value| value.trim().is_empty()) {
        return Err("enmaas_policy: qos_clusters must contain at least one non-empty name".into());
    }
    if cfg.header_prefix.chars().any(char::is_control) {
        return Err("enmaas_policy: header_prefix must not contain control characters".into());
    }
    cfg.promoted_model_header_name()?;
    cfg.metering_auth_token()?;
    cfg.response_class_header_name()?;
    cfg.credential_header_name()?;
    Ok(())
}

fn validate_bucket(name: &str, burst: u32, refill_per_second: f64) -> Result<(), FilterError> {
    if burst > 0 && (!refill_per_second.is_finite() || refill_per_second <= 0.0) {
        return Err(format!("enmaas_policy: {name}_refill_per_second must be positive when burst is set").into());
    }
    if burst == 0 && refill_per_second != 0.0 {
        return Err(format!("enmaas_policy: {name}_refill_per_second must be zero when burst is disabled").into());
    }
    Ok(())
}

fn read_bearer_token(path: Option<&str>, field: &str) -> Result<Option<HeaderValue>, FilterError> {
    let Some(path) = path else {
        return Ok(None);
    };
    let token = fs::read_to_string(path)
        .map_err(|error| -> FilterError { format!("enmaas_policy: {field} could not be read: {error}").into() })?;
    let token = token.trim();
    if token.is_empty() {
        return Err(format!("enmaas_policy: {field} is empty").into());
    }
    if token.chars().any(char::is_whitespace) {
        return Err(format!("enmaas_policy: {field} contains whitespace").into());
    }
    let value = format!("Bearer {token}");
    HeaderValue::from_str(&value)
        .map(Some)
        .map_err(|error| format!("enmaas_policy: {field} is invalid: {error}").into())
}

fn default_timeout_seconds() -> u64 {
    DEFAULT_TIMEOUT_SECONDS
}

fn default_true() -> bool {
    true
}

fn default_allow_missing_model() -> bool {
    true
}

fn default_decision_cache_ttl_seconds() -> u64 {
    DEFAULT_DECISION_CACHE_TTL_SECONDS
}

fn default_decision_cache_max_entries() -> usize {
    DEFAULT_DECISION_CACHE_MAX_ENTRIES
}

fn default_username_metadata() -> String {
    DEFAULT_USERNAME_METADATA.to_owned()
}

fn default_group_metadata() -> String {
    DEFAULT_GROUP_METADATA.to_owned()
}

fn default_subscription_metadata() -> String {
    DEFAULT_SUBSCRIPTION_METADATA.to_owned()
}

fn default_model_metadata() -> String {
    DEFAULT_MODEL_METADATA.to_owned()
}

fn default_identity_model_metadata() -> String {
    DEFAULT_IDENTITY_MODEL_METADATA.to_owned()
}

fn default_promoted_model_header() -> String {
    DEFAULT_PROMOTED_MODEL_HEADER.to_owned()
}

fn default_class_metadata() -> String {
    DEFAULT_CLASS_METADATA.to_owned()
}

fn default_subject_metadata() -> String {
    DEFAULT_SUBJECT_METADATA.to_owned()
}

fn default_retry_after_metadata() -> String {
    DEFAULT_RETRY_AFTER_METADATA.to_owned()
}

fn default_key_alias_metadata() -> String {
    DEFAULT_KEY_ALIAS_METADATA.to_owned()
}

fn default_class() -> String {
    DEFAULT_CLASS.to_owned()
}

fn default_metering_url() -> String {
    DEFAULT_METERING_URL.to_owned()
}

fn default_feature_key() -> String {
    "inference-tokens".to_owned()
}

fn default_key_alias_premium() -> String {
    "premium".to_owned()
}

fn default_key_alias_main() -> String {
    "main".to_owned()
}

fn default_key_alias_ratelimit() -> String {
    "main".to_owned()
}

fn default_source() -> String {
    DEFAULT_SOURCE.to_owned()
}

fn default_model() -> String {
    DEFAULT_MODEL.to_owned()
}

fn default_response_class_header() -> String {
    DEFAULT_RESPONSE_CLASS_HEADER.to_owned()
}

fn default_main_burst() -> u32 {
    10
}

fn default_main_refill() -> f64 {
    1.0
}

fn default_max_keys() -> usize {
    100_000
}

fn default_key_clusters() -> Vec<String> {
    vec!["cb-glm".to_owned()]
}

fn default_qos_clusters() -> Vec<String> {
    vec!["cb-glm".to_owned()]
}

fn default_credential_header() -> String {
    DEFAULT_CREDENTIAL_HEADER.to_owned()
}

fn default_header_prefix() -> String {
    DEFAULT_HEADER_PREFIX.to_owned()
}

fn default_key_env() -> String {
    DEFAULT_KEY_ENV.to_owned()
}
