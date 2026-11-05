#![allow(clippy::missing_docs_in_private_items, reason = "private policy protocol types")]

use std::time::Instant;

use serde::Deserialize;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum DenyReason {
    QuotaExhausted,
    ModelNotAllowed,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct PolicyDecision {
    pub allowed: bool,
    #[serde(rename = "denyReason", default)]
    pub deny_reason: Option<DenyReason>,
    #[serde(rename = "qosClass", default = "default_class")]
    pub qos_class: String,
    #[serde(rename = "qosSubject", default)]
    pub qos_subject: Option<String>,
    #[serde(rename = "retryAfterSeconds", default)]
    pub retry_after_seconds: Option<u64>,
    #[serde(rename = "qosKeyAlias", default)]
    pub qos_key_alias: Option<String>,
    #[serde(rename = "cacheTtlSeconds", default)]
    pub cache_ttl_seconds: Option<u64>,
}

impl PolicyDecision {
    pub(crate) fn fail_open(default_class: &str, subject: &str) -> Self {
        Self {
            allowed: true,
            deny_reason: None,
            qos_class: default_class.to_owned(),
            qos_subject: Some(subject.to_owned()),
            retry_after_seconds: None,
            qos_key_alias: None,
            cache_ttl_seconds: None,
        }
    }

    pub(crate) fn normalized_class(&self, default_class: &str) -> String {
        match self.qos_class.as_str() {
            "premium" | "main" | "ratelimit" => self.qos_class.clone(),
            _ => default_class.to_owned(),
        }
    }
}

pub(crate) struct CachedDecision {
    pub decision: PolicyDecision,
    pub expires_at: Instant,
}

fn default_class() -> String {
    "main".to_owned()
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct MeteringEntitlement {
    pub has_access: bool,
    #[serde(default = "default_true")]
    pub model_allowed: bool,
    #[serde(default)]
    pub allowed: Option<bool>,
    #[serde(default)]
    pub deny_reason: Option<String>,
    #[serde(default)]
    pub quota_retry_after: Option<u64>,
    #[serde(default)]
    pub qos_class: Option<String>,
    #[serde(default)]
    pub qos_subject: Option<String>,
    #[serde(default)]
    pub qos_retry_after: Option<u64>,
    #[serde(default)]
    pub qos_key_alias: Option<String>,
}

impl MeteringEntitlement {
    pub(crate) fn into_decision(
        self,
        default_class: &str,
        premium_alias: &str,
        main_alias: &str,
        ratelimit_alias: &str,
    ) -> PolicyDecision {
        let qos_class = self.qos_class.unwrap_or_else(|| default_class.to_owned());
        let deny_reason = self.deny_reason.as_deref().and_then(DenyReason::parse).or_else(|| {
            if !self.model_allowed {
                Some(DenyReason::ModelNotAllowed)
            } else if !self.has_access {
                Some(DenyReason::QuotaExhausted)
            } else {
                None
            }
        });
        let allowed = self.allowed.unwrap_or(self.has_access && self.model_allowed);
        let retry_after_seconds = match deny_reason {
            Some(DenyReason::QuotaExhausted) => self.quota_retry_after,
            _ if qos_class == "ratelimit" => self.qos_retry_after.or(Some(30)),
            _ => None,
        };
        let qos_key_alias = self.qos_key_alias.or_else(|| {
            Some(
                match qos_class.as_str() {
                    "premium" => premium_alias,
                    "ratelimit" => ratelimit_alias,
                    _ => main_alias,
                }
                .to_owned(),
            )
        });

        PolicyDecision {
            allowed,
            deny_reason,
            qos_class,
            qos_subject: self.qos_subject,
            retry_after_seconds,
            qos_key_alias,
            cache_ttl_seconds: None,
        }
    }
}

impl DenyReason {
    fn parse(value: &str) -> Option<Self> {
        match value {
            "quota_exhausted" => Some(Self::QuotaExhausted),
            "model_not_allowed" => Some(Self::ModelNotAllowed),
            _ => None,
        }
    }
}

fn default_true() -> bool {
    true
}
