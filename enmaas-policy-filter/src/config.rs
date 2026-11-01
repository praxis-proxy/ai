#![allow(
    clippy::missing_docs_in_private_items,
    clippy::struct_excessive_bools,
    reason = "private configuration internals"
)]

use std::{fs, time::Duration};

use http::{HeaderName, HeaderValue};
use praxis_filter::FilterError;
use serde::Deserialize;

const DEFAULT_TIMEOUT_SECONDS: u64 = 2;
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

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum QosEnforcement {
    #[default]
    Soft,
    Off,
    Hard,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct EnmaasPolicyConfig {
    pub policy_url: String,

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

    #[serde(default)]
    pub internal_auth_file: Option<String>,

    #[serde(default = "default_true")]
    pub model_access_enabled: bool,

    #[serde(default = "default_true")]
    pub allow_missing_model: bool,
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

    pub(crate) fn internal_auth_token(&self) -> Result<Option<HeaderValue>, FilterError> {
        let Some(path) = self.internal_auth_file.as_deref() else {
            return Ok(None);
        };
        let token = fs::read_to_string(path).map_err(|error| -> FilterError {
            format!("enmaas_policy: internal_auth_file could not be read: {error}").into()
        })?;
        let token = token.trim();
        if token.is_empty() {
            return Err("enmaas_policy: internal_auth_file is empty".into());
        }
        if token.chars().any(char::is_whitespace) {
            return Err("enmaas_policy: internal_auth_file contains whitespace".into());
        }
        let value = format!("Bearer {token}");
        HeaderValue::from_str(&value)
            .map(Some)
            .map_err(|error| -> FilterError { format!("enmaas_policy: internal_auth_file is invalid: {error}").into() })
    }
}

pub(crate) fn validate_config(cfg: &EnmaasPolicyConfig) -> Result<(), FilterError> {
    if cfg.policy_url.trim().is_empty() {
        return Err("enmaas_policy: policy_url must not be empty".into());
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
    ] {
        if value.trim().is_empty() {
            return Err(format!("enmaas_policy: {name} must not be empty").into());
        }
    }
    if !matches!(cfg.default_class.as_str(), "premium" | "main" | "ratelimit") {
        return Err("enmaas_policy: default_class must be premium, main, or ratelimit".into());
    }
    cfg.promoted_model_header_name()?;
    cfg.internal_auth_token()?;
    Ok(())
}

fn default_timeout_seconds() -> u64 {
    DEFAULT_TIMEOUT_SECONDS
}

fn default_true() -> bool {
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

fn default_class() -> String {
    DEFAULT_CLASS.to_owned()
}

fn default_key_alias_metadata() -> String {
    DEFAULT_KEY_ALIAS_METADATA.to_owned()
}
