#![allow(
    clippy::missing_docs_in_private_items,
    clippy::struct_excessive_bools,
    clippy::too_many_arguments,
    clippy::significant_drop_tightening,
    clippy::too_many_lines,
    reason = "private filter internals"
)]

use std::{
    borrow::Cow,
    collections::HashMap,
    env,
    sync::Arc,
    time::{Duration, Instant},
};

use async_trait::async_trait;
use bytes::Bytes;
use http::{
    HeaderName, HeaderValue,
    header::{AUTHORIZATION, CONTENT_TYPE},
};
use metrics::{counter, histogram};
use praxis_ai_apis::{
    callout_target::AddressPolicy,
    subrequest::{self, SubRequest, SubRequestClient, SubRequestError, SubResponse},
};
use praxis_core::subrequest::SubRequestConnector;
use praxis_filter::{
    BodyAccess, BodyMode, FilterAction, FilterError, HttpFilter, HttpFilterContext, PendingHeaderResult, Rejection,
    parse_filter_config,
};
use tokio::sync::RwLock;
use tracing::{debug, warn};

use crate::{
    buckets::{BucketLimits, BucketOutcome, LocalBuckets},
    config::{CredentialMode, EnmaasPolicyConfig, QosEnforcement, validate_config},
    events::{EventReporter, RequestState, store_state, take_state, update_response},
    keys::{KeyRing, SelectedKey},
    policy::{CachedDecision, DenyReason, PolicyDecision, PolicyIdentity, PolicyRequest},
};

const MAX_POLICY_RESPONSE_BYTES: usize = 64 * 1024;
const STATUS_BUDGET_EXHAUSTED: u16 = 429;
const STATUS_MODEL_DENIED: u16 = 403;
const STATUS_POLICY_UNAVAILABLE: u16 = 503;
const STATUS_RATE_LIMITED: u16 = 429;

pub(crate) struct EnmaasPolicyFilter {
    config: EnmaasPolicyConfig,
    address_policy: AddressPolicy,
    client: SubRequestClient,
    policy_auth_token: Option<HeaderValue>,
    promoted_model_header: HeaderName,
    response_class_header: Option<HeaderName>,
    credential_header: HeaderName,
    key_clusters: Vec<String>,
    keys: KeyRing,
    events: EventReporter,
    buckets: LocalBuckets,
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
        let policy_auth_token = config.policy_auth_token()?;
        let metering_auth_token = config.metering_auth_token()?;
        let promoted_model_header = config.promoted_model_header_name()?;
        let response_class_header = config.response_class_header_name()?;
        let credential_header = config.credential_header_name()?;
        let client = isolated_subrequest_client(4);
        let keys = if config.credential_mode == CredentialMode::Off {
            KeyRing::default()
        } else {
            let default_key = env::var(&config.default_key_env)
                .ok()
                .filter(|value| !value.trim().is_empty())
                .map(|value| format!("{}{}", config.header_prefix, value.trim()));
            KeyRing::load(config.key_file.as_deref(), &config.header_prefix, default_key)?
        };
        let events = EventReporter::new(
            client.clone(),
            &config.metering_url,
            config.timeout(),
            address_policy,
            metering_auth_token,
            config.source.clone(),
            config.provider.clone(),
            config.default_model.clone(),
        );
        let buckets = LocalBuckets::new(
            BucketLimits::new(config.main_burst, config.main_refill_per_second),
            BucketLimits::new(config.premium_burst, config.premium_refill_per_second),
            config.max_keys,
        );

        Ok(Box::new(Self {
            key_clusters: config.key_clusters.clone(),
            config,
            address_policy,
            client,
            policy_auth_token,
            promoted_model_header,
            response_class_header,
            credential_header,
            keys,
            events,
            buckets,
            cache: Arc::new(RwLock::new(HashMap::new())),
        }))
    }

    fn resolve_model(&self, ctx: &HttpFilterContext<'_>) -> String {
        if let Some(model) = ctx.get_metadata(&self.config.model_metadata) {
            return model.to_owned();
        }
        if let Some(model) = ctx
            .buffered_request_body
            .as_ref()
            .and_then(|body| serde_json::from_slice::<serde_json::Value>(body).ok())
            .and_then(|value| {
                value
                    .get("model")
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_owned)
            })
        {
            return model;
        }
        if let Some(model) = ctx.get_metadata(&self.config.identity_model_metadata) {
            return model.to_owned();
        }
        if let Ok(PendingHeaderResult::Value(model)) = ctx.pending_header_value(&self.promoted_model_header) {
            return model;
        }
        String::new()
    }

    fn applies_to_key_cluster(&self, ctx: &HttpFilterContext<'_>) -> bool {
        ctx.cluster
            .as_deref()
            .is_some_and(|cluster| self.key_clusters.iter().any(|candidate| candidate == cluster))
    }

    async fn cached(&self, key: &str) -> Option<PolicyDecision> {
        if self.config.decision_cache_ttl_seconds == 0 {
            return None;
        }
        let mut cache = self.cache.write().await;
        let entry = cache.get(key)?;
        if Instant::now() < entry.expires_at {
            counter!("enmaas_policy_cache_requests_total", "result" => "hit").increment(1);
            Some(entry.decision.clone())
        } else {
            cache.remove(key);
            counter!("enmaas_policy_cache_requests_total", "result" => "expired").increment(1);
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
        if let Some(token) = &self.policy_auth_token {
            headers.insert(AUTHORIZATION, token.clone());
        }

        let request = SubRequest {
            method: http::Method::POST,
            uri: http::Uri::default(),
            headers,
            body: Bytes::from(body),
        };

        let started = Instant::now();
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
        histogram!("enmaas_policy_callout_duration_seconds").record(started.elapsed().as_secs_f64());

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
        cached: bool,
    ) -> FilterAction {
        let class = decision.normalized_class(&self.config.default_class);
        let subject = decision
            .qos_subject
            .as_deref()
            .filter(|value| !value.is_empty())
            .unwrap_or(username)
            .to_owned();
        ctx.set_metadata(self.config.class_metadata.as_str(), class.clone());
        ctx.set_metadata(self.config.subject_metadata.as_str(), subject.clone());
        if let Some(retry_after) = decision.retry_after_seconds {
            ctx.set_metadata(self.config.retry_after_metadata.as_str(), retry_after.to_string());
        }
        if let Some(alias) = &decision.qos_key_alias {
            ctx.set_metadata(self.config.key_alias_metadata.as_str(), alias.clone());
        }

        let reason = if decision.allowed {
            "none"
        } else {
            match decision.deny_reason {
                Some(DenyReason::QuotaExhausted) => "quota_exhausted",
                Some(DenyReason::ModelNotAllowed) => "model_not_allowed",
                None => "policy_unavailable",
            }
        };
        counter!(
            "enmaas_policy_decisions_total",
            "result" => if decision.allowed { "allow" } else { "deny" },
            "reason" => reason,
            "class" => class.clone(),
            "cached" => if cached { "true" } else { "false" },
        )
        .increment(1);

        if !decision.allowed {
            return match decision.deny_reason {
                Some(DenyReason::QuotaExhausted) => reject_budget(&class, decision.retry_after_seconds),
                Some(DenyReason::ModelNotAllowed) => reject_model(&class),
                None => reject_unavailable("policy decision missing deny reason"),
            };
        }

        if class == "ratelimit" {
            match self.config.qos_enforcement {
                QosEnforcement::Hard => {
                    counter!(
                        "enmaas_policy_enforcement_total",
                        "class" => class.clone(),
                        "result" => "denied",
                    )
                    .increment(1);
                    return reject_ratelimit(&class, decision.retry_after_seconds);
                },
                QosEnforcement::Soft | QosEnforcement::Off => {
                    counter!(
                        "enmaas_policy_enforcement_total",
                        "class" => class.clone(),
                        "result" => "soft",
                    )
                    .increment(1);
                },
            }
        } else if self.config.qos_enforcement == QosEnforcement::Hard {
            match self.buckets.admit(&class, &subject) {
                BucketOutcome::Admitted => {
                    counter!(
                        "enmaas_policy_enforcement_total",
                        "class" => class.clone(),
                        "result" => "admitted",
                    )
                    .increment(1);
                },
                BucketOutcome::Rejected { retry_after } => {
                    counter!(
                        "enmaas_policy_enforcement_total",
                        "class" => class.clone(),
                        "result" => "denied",
                    )
                    .increment(1);
                    return reject_ratelimit(&class, Some(retry_after));
                },
            }
        } else {
            counter!(
                "enmaas_policy_enforcement_total",
                "class" => class.clone(),
                "result" => "soft",
            )
            .increment(1);
        }

        if let Some(action) = self.inject_key(ctx, decision, &class, &subject) {
            return action;
        }

        FilterAction::Continue
    }

    fn inject_key(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        decision: &PolicyDecision,
        class: &str,
        subject: &str,
    ) -> Option<FilterAction> {
        if self.config.credential_mode == CredentialMode::Off || !self.applies_to_key_cluster(ctx) {
            return None;
        }
        let (alias, shadow) = match self.config.credential_mode {
            CredentialMode::Off => return None,
            CredentialMode::Shadow => (Some("main"), true),
            CredentialMode::Active => (decision.qos_key_alias.as_deref().or(Some(class)), false),
        };
        let selected = self.keys.resolve(alias, subject).or_else(|| self.keys.default_key());
        let Some(SelectedKey { id, header_value }) = selected else {
            return Some(reject_unavailable("no provider credential available"));
        };
        if HeaderValue::from_str(&header_value).is_err() {
            return Some(reject_unavailable("selected provider credential is invalid"));
        }
        ctx.extra_request_headers
            .push((Cow::Owned(self.credential_header.as_str().to_owned()), header_value));
        counter!(
            "enmaas_policy_key_selected_total",
            "requested_class" => class.to_owned(),
            "key_id" => id,
            "shadow" => shadow.to_string(),
        )
        .increment(1);
        None
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
            return Ok(reject_unavailable("identity is unavailable"));
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
            return Ok(reject_model(&self.config.default_class));
        }
        let path = ctx.request.uri.path().to_owned();
        let cache_key = format!("{username}\u{1f}{model}\u{1f}{path}");
        let start = ctx.request_start;
        let user_agent = ctx
            .request
            .headers
            .get(http::header::USER_AGENT)
            .and_then(|value| value.to_str().ok())
            .unwrap_or_default()
            .to_owned();

        let action = if let Some(decision) = self.cached(&cache_key).await {
            self.apply_decision(ctx, &decision, &username, true)
        } else {
            counter!("enmaas_policy_cache_requests_total", "result" => "miss").increment(1);
            match self
                .fetch_decision(&username, &group, &subscription, &model, &path)
                .await
            {
                Ok(decision) => {
                    self.store(cache_key, decision.clone()).await;
                    self.apply_decision(ctx, &decision, &username, false)
                },
                Err(error) => {
                    counter!("enmaas_policy_callout_failures_total", "reason" => "callout").increment(1);
                    warn!(%error, "enmaas_policy: policy callout failed");
                    if self.config.fail_open {
                        let decision = PolicyDecision::fail_open(&self.config.default_class, &username);
                        self.apply_decision(ctx, &decision, &username, false)
                    } else {
                        reject_unavailable("policy service unavailable")
                    }
                },
            }
        };

        if matches!(action, FilterAction::Continue) {
            store_state(
                ctx,
                RequestState {
                    username,
                    group,
                    subscription,
                    model,
                    user_agent,
                    start,
                    is_error: false,
                    response_status: 0,
                },
            );
        }

        Ok(action)
    }

    fn response_body_access(&self) -> BodyAccess {
        BodyAccess::ReadOnly
    }

    fn response_body_mode(&self) -> BodyMode {
        BodyMode::Stream
    }

    async fn on_response(&self, ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        let status = ctx
            .response_header
            .as_ref()
            .map_or(0, |response| response.status.as_u16());
        let is_error = ctx
            .response_header
            .as_ref()
            .is_some_and(|response| !response.status.is_success());
        update_response(ctx, status, is_error);

        let Some(header_name) = &self.response_class_header else {
            return Ok(FilterAction::Continue);
        };
        let Some(class) = ctx.get_metadata(&self.config.class_metadata).map(str::to_owned) else {
            return Ok(FilterAction::Continue);
        };
        let Some(response) = ctx.response_header.as_mut() else {
            return Ok(FilterAction::Continue);
        };
        let Ok(value) = HeaderValue::from_str(&class) else {
            return Ok(FilterAction::Continue);
        };
        response.headers.insert(header_name.clone(), value);
        ctx.response_headers_modified = true;
        Ok(FilterAction::Continue)
    }

    fn on_response_body(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        _body: &mut Option<Bytes>,
        end_of_stream: bool,
    ) -> Result<FilterAction, FilterError> {
        if !end_of_stream {
            return Ok(FilterAction::Continue);
        }
        if let Some(state) = take_state(ctx) {
            self.events.report(ctx, &state);
        }
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
        STATUS_RATE_LIMITED,
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
            .with_header("X-EnMaaS-QoS-Class", class.to_owned())
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
            .with_header("X-EnMaaS-QoS-Class", class.to_owned())
            .with_body(Bytes::from(body.to_string())),
    )
}

fn reject_unavailable(message: &str) -> FilterAction {
    let body = serde_json::json!({
        "error": {
            "message": message,
            "type": "service_unavailable",
            "code": "policy_unavailable"
        }
    });
    FilterAction::Reject(
        Rejection::status(STATUS_POLICY_UNAVAILABLE)
            .with_header("Content-Type", "application/json")
            .with_body(Bytes::from(body.to_string())),
    )
}
