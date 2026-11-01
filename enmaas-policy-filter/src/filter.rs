#![allow(
    clippy::missing_docs_in_private_items,
    clippy::struct_excessive_bools,
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "private filter internals"
)]

use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, Instant},
};

use async_trait::async_trait;
use bytes::Bytes;
use http::{
    HeaderValue,
    header::{AUTHORIZATION, CONTENT_TYPE},
};
use praxis_ai_apis::{
    callout_target::AddressPolicy,
    subrequest::{self, SubRequest, SubRequestClient, SubRequestError, SubResponse},
};
use praxis_core::subrequest::SubRequestConnector;
use praxis_filter::{
    FilterAction, FilterError, HttpFilter, HttpFilterContext, PendingHeaderResult, Rejection, parse_filter_config,
};
use tokio::sync::RwLock;
use tracing::{debug, warn};

use crate::{
    config::{EnmaasPolicyConfig, validate_config},
    policy::{CachedDecision, PolicyDecision, PolicyIdentity, PolicyRequest},
};

const MAX_POLICY_RESPONSE_BYTES: usize = 64 * 1024;
const STATUS_BUDGET_EXHAUSTED: u16 = 429;
const STATUS_MODEL_DENIED: u16 = 403;
const STATUS_POLICY_UNAVAILABLE: u16 = 503;
const QOS_CLASS_HEADER: &str = "x-enmaas-qos-class";

pub(crate) struct EnmaasPolicyFilter {
    config: EnmaasPolicyConfig,
    address_policy: AddressPolicy,
    client: SubRequestClient,
    internal_auth_token: Option<HeaderValue>,
    promoted_model_header: http::HeaderName,
    cache: Arc<RwLock<HashMap<String, CachedDecision>>>,
}

impl EnmaasPolicyFilter {
    pub(crate) fn from_config(config: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
        let config: EnmaasPolicyConfig = parse_filter_config("enmaas_policy", config)?;
        validate_config(&config)?;
        let address_policy = if config.allow_private_endpoint {
            AddressPolicy::AllowPrivate
        } else {
            AddressPolicy::PublicOnly
        };
        let internal_auth_token = config.internal_auth_token()?;
        let promoted_model_header = config.promoted_model_header_name()?;
        let client = isolated_subrequest_client(4);

        Ok(Box::new(Self {
            config,
            address_policy,
            client,
            internal_auth_token,
            promoted_model_header,
            cache: Arc::new(RwLock::new(HashMap::new())),
        }))
    }

    fn resolve_model(&self, ctx: &HttpFilterContext<'_>) -> String {
        if let Some(model) = ctx.get_metadata(&self.config.model_metadata) {
            return model.to_owned();
        }
        if let Some(model) = ctx.get_metadata(&self.config.identity_model_metadata) {
            return model.to_owned();
        }
        if let Ok(PendingHeaderResult::Value(model)) = ctx.pending_header_value(&self.promoted_model_header) {
            return model;
        }
        String::new()
    }

    async fn cached(&self, key: &str) -> Option<PolicyDecision> {
        if self.config.decision_cache_ttl_seconds == 0 {
            return None;
        }
        let mut cache = self.cache.write().await;
        let entry = cache.get(key)?;
        if Instant::now() < entry.expires_at {
            Some(entry.decision.clone())
        } else {
            cache.remove(key);
            None
        }
    }

    async fn store(&self, key: String, decision: PolicyDecision) {
        if self.config.decision_cache_ttl_seconds == 0 {
            return;
        }
        let mut cache = self.cache.write().await;
        if cache.len() >= self.config.decision_cache_max_entries {
            cache.clear();
        }
        let default_ttl = self.config.decision_cache_ttl_seconds;
        let ttl = decision.cache_ttl_seconds.unwrap_or(default_ttl).max(1);
        cache.insert(
            key,
            CachedDecision {
                decision,
                expires_at: Instant::now() + Duration::from_secs(ttl),
            },
        );
    }

    async fn fetch_decision(
        &self,
        username: &str,
        group: &str,
        subscription: &str,
        model: &str,
        path: &str,
    ) -> Result<PolicyDecision, String> {
        let request = PolicyRequest {
            version: 1,
            identity: PolicyIdentity {
                username,
                group,
                subscription,
            },
            model,
            path,
        };
        let body = serde_json::to_vec(&request).map_err(|error| format!("request serialization failed: {error}"))?;

        let mut headers = http::HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        if let Some(token) = &self.internal_auth_token {
            headers.insert(AUTHORIZATION, token.clone());
        }

        let request = SubRequest {
            method: http::Method::POST,
            uri: http::Uri::default(),
            headers,
            body: Bytes::from(body),
        };

        let response: SubResponse = subrequest::execute_configured_url(
            &self.client,
            &self.config.policy_url,
            request,
            MAX_POLICY_RESPONSE_BYTES,
            self.config.timeout(),
            self.address_policy,
        )
        .await
        .map_err(|error| describe_subrequest_error(&error))?;

        if !(200..300).contains(&response.status) {
            return Err(format!("policy service returned HTTP {}", response.status));
        }

        let decision: PolicyDecision = serde_json::from_slice(&response.body)
            .map_err(|error| format!("policy response was not valid JSON: {error}"))?;
        if decision.version != 1 {
            return Err(format!("unsupported policy version {}", decision.version));
        }
        Ok(decision)
    }

    fn apply_decision(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        decision: &PolicyDecision,
        username: &str,
    ) -> FilterAction {
        let class = decision.normalized_class(&self.config.default_class);
        let subject = decision
            .qos_subject
            .as_deref()
            .filter(|value| !value.is_empty())
            .unwrap_or(username);
        ctx.set_metadata(self.config.class_metadata.as_str(), class.clone());
        ctx.set_metadata(self.config.subject_metadata.as_str(), subject.to_owned());
        if let Some(retry_after) = decision.qos_retry_after {
            ctx.set_metadata(self.config.retry_after_metadata.as_str(), retry_after.to_string());
        }
        if let Some(alias) = &decision.qos_key_alias {
            ctx.set_metadata(self.config.key_alias_metadata.as_str(), alias.clone());
        }

        if !decision.has_access {
            return reject_budget(&class, decision.qos_retry_after);
        }
        if self.config.model_access_enabled && !decision.model_allowed {
            return reject_model(&class);
        }
        if class == "ratelimit" && self.config.qos_enforcement == crate::config::QosEnforcement::Hard {
            return reject_ratelimit(&class, decision.qos_retry_after);
        }

        FilterAction::Continue
    }
}

#[async_trait]
impl HttpFilter for EnmaasPolicyFilter {
    fn name(&self) -> &'static str {
        "enmaas_policy"
    }

    async fn on_request(&self, ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        if !self.config.enabled {
            return Ok(FilterAction::Continue);
        }

        let username = ctx
            .get_metadata(&self.config.username_metadata)
            .or(self.config.default_username.as_deref())
            .unwrap_or_default()
            .to_owned();
        if username.is_empty() {
            if self.config.fail_open {
                debug!("enmaas_policy: no identity; failing open");
                return Ok(FilterAction::Continue);
            }
            return Ok(reject_unavailable());
        }

        let group = ctx
            .get_metadata(&self.config.group_metadata)
            .unwrap_or_default()
            .to_owned();
        let subscription = ctx
            .get_metadata(&self.config.subscription_metadata)
            .unwrap_or_default()
            .to_owned();
        let model = self.resolve_model(ctx);
        if model.is_empty() && !self.config.allow_missing_model {
            let class = self.config.default_class.clone();
            return Ok(reject_model(&class));
        }
        let path = ctx.request.uri.path().to_owned();
        let cache_key = format!("{username}\u{1f}{model}\u{1f}{path}");

        if let Some(decision) = self.cached(&cache_key).await {
            return Ok(self.apply_decision(ctx, &decision, &username));
        }

        match self
            .fetch_decision(&username, &group, &subscription, &model, &path)
            .await
        {
            Ok(decision) => {
                self.store(cache_key, decision.clone()).await;
                Ok(self.apply_decision(ctx, &decision, &username))
            },
            Err(error) => {
                warn!(%error, "enmaas_policy: policy callout failed");
                if self.config.fail_open {
                    let decision = PolicyDecision::fail_open(&self.config.default_class, &username);
                    Ok(self.apply_decision(ctx, &decision, &username))
                } else {
                    Ok(reject_unavailable())
                }
            },
        }
    }

    async fn on_response(&self, ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        let Some(class) = ctx.get_metadata(&self.config.class_metadata).map(str::to_owned) else {
            return Ok(FilterAction::Continue);
        };
        let Some(response) = ctx.response_header.as_mut() else {
            return Ok(FilterAction::Continue);
        };
        let Ok(value) = HeaderValue::from_str(&class) else {
            return Ok(FilterAction::Continue);
        };
        response
            .headers
            .insert(http::HeaderName::from_static(QOS_CLASS_HEADER), value);
        ctx.response_headers_modified = true;
        Ok(FilterAction::Continue)
    }
}

fn isolated_subrequest_client(pool_size: usize) -> SubRequestClient {
    praxis_tls::provider::install();
    SubRequestClient::new(SubRequestConnector::new(pool_size, None))
}

fn describe_subrequest_error(error: &SubRequestError) -> String {
    format!("policy callout failed: {error}")
}

fn reject_budget(class: &str, retry_after: Option<u64>) -> FilterAction {
    reject(
        STATUS_BUDGET_EXHAUSTED,
        class,
        retry_after.unwrap_or(30).max(1),
        "token budget exhausted",
        "rate_limit_error",
        "token_budget_exhausted",
    )
}

fn reject_ratelimit(class: &str, retry_after: Option<u64>) -> FilterAction {
    reject(
        STATUS_BUDGET_EXHAUSTED,
        class,
        retry_after.unwrap_or(30).max(1),
        "request rate exceeded for this QoS class",
        "rate_limit_error",
        "qos_rate_limited",
    )
}

fn reject_model(class: &str) -> FilterAction {
    let body = serde_json::json!({
        "error": {
            "message": "model is not authorized for this user",
            "type": "permission_error",
            "code": "model_not_allowed"
        }
    });
    FilterAction::Reject(
        Rejection::status(STATUS_MODEL_DENIED)
            .with_header("Content-Type", "application/json")
            .with_header(QOS_CLASS_HEADER, class.to_owned())
            .with_body(Bytes::from(body.to_string())),
    )
}

fn reject(status: u16, class: &str, retry_after: u64, message: &str, error_type: &str, code: &str) -> FilterAction {
    let body = serde_json::json!({
        "error": {
            "message": message,
            "type": error_type,
            "code": code
        }
    });
    FilterAction::Reject(
        Rejection::status(status)
            .with_header("Content-Type", "application/json")
            .with_header("Retry-After", retry_after.to_string())
            .with_header(QOS_CLASS_HEADER, class.to_owned())
            .with_body(Bytes::from(body.to_string())),
    )
}

fn reject_unavailable() -> FilterAction {
    FilterAction::Reject(
        Rejection::status(STATUS_POLICY_UNAVAILABLE)
            .with_header("Content-Type", "application/json")
            .with_body(Bytes::from_static(br#"{"error":{"message":"policy service unavailable","type":"service_unavailable","code":"policy_unavailable"}}"#)),
    )
}
