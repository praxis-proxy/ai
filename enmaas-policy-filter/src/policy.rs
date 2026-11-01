#![allow(clippy::missing_docs_in_private_items, reason = "private policy protocol types")]

use std::time::Instant;

use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize)]
pub(crate) struct PolicyRequest<'a> {
    pub version: u8,
    pub identity: PolicyIdentity<'a>,
    pub model: &'a str,
    pub path: &'a str,
}

#[derive(Debug, Serialize)]
pub(crate) struct PolicyIdentity<'a> {
    pub username: &'a str,
    pub group: &'a str,
    pub subscription: &'a str,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct PolicyDecision {
    pub version: u32,
    #[serde(rename = "hasAccess")]
    pub has_access: bool,
    #[serde(rename = "modelAllowed", default = "default_true")]
    pub model_allowed: bool,
    #[serde(rename = "qosClass", default = "default_class")]
    pub qos_class: String,
    #[serde(rename = "qosSubject", default)]
    pub qos_subject: Option<String>,
    #[serde(rename = "qosRetryAfter", default)]
    pub qos_retry_after: Option<u64>,
    #[serde(rename = "qosKeyAlias", default)]
    pub qos_key_alias: Option<String>,
    #[serde(rename = "cacheTtlSeconds", default)]
    pub cache_ttl_seconds: Option<u64>,
}

impl PolicyDecision {
    pub(crate) fn fail_open(default_class: &str, subject: &str) -> Self {
        Self {
            version: 1,
            has_access: true,
            model_allowed: true,
            qos_class: default_class.to_owned(),
            qos_subject: Some(subject.to_owned()),
            qos_retry_after: None,
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

fn default_true() -> bool {
    true
}

fn default_class() -> String {
    "main".to_owned()
}
