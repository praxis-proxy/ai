#![allow(
    clippy::missing_docs_in_private_items,
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "private event internals"
)]

use std::time::Instant;

use bytes::Bytes;
use http::{
    HeaderValue,
    header::{AUTHORIZATION, CONTENT_TYPE},
};
use metrics::counter;
use praxis_ai_apis::{
    callout_target::AddressPolicy,
    subrequest::{self, SubRequest, SubRequestClient},
};
use praxis_filter::HttpFilterContext;
use serde_json::json;
use tracing::{trace, warn};

const CE_SPEC_VERSION: &str = "1.0";
const CE_TYPE_ERROR: &str = "inference.request.error";
const CE_TYPE_USAGE: &str = "inference.tokens.used";
const MAX_RESPONSE_BYTES: usize = 64 * 1024;
const META_TOKEN_INPUT: &str = "token.input";
const META_TOKEN_OUTPUT: &str = "token.output";
const META_TOKEN_TOTAL: &str = "token.total";
const META_TOKEN_CACHE_READ: &str = "token.cache_read";
const META_TOKEN_CACHE_WRITE: &str = "token.cache_write";
const META_TOKEN_REASONING: &str = "token.reasoning";

pub(crate) struct RequestState {
    pub username: String,
    pub group: String,
    pub subscription: String,
    pub model: String,
    pub user_agent: String,
    pub start: Instant,
    pub is_error: bool,
    pub response_status: u16,
}

#[derive(Clone)]
pub(crate) struct EventReporter {
    client: SubRequestClient,
    url: String,
    timeout: std::time::Duration,
    address_policy: AddressPolicy,
    auth_token: Option<HeaderValue>,
    source: String,
    provider: Option<String>,
    default_model: String,
}

impl EventReporter {
    pub(crate) fn new(
        client: SubRequestClient,
        metering_url: &str,
        timeout: std::time::Duration,
        address_policy: AddressPolicy,
        auth_token: Option<HeaderValue>,
        source: String,
        provider: Option<String>,
        default_model: String,
    ) -> Self {
        Self {
            client,
            url: format!("{}/api/v1/events", metering_url.trim_end_matches('/')),
            timeout,
            address_policy,
            auth_token,
            source,
            provider,
            default_model,
        }
    }

    pub(crate) fn report(&self, ctx: &HttpFilterContext<'_>, state: &RequestState) {
        let event_id = ctx.id_generator.generate(ctx.time_source);
        let provider = self
            .provider
            .as_deref()
            .or_else(|| ctx.cluster_name())
            .unwrap_or("unknown");
        let model = if state.model.is_empty() {
            &self.default_model
        } else {
            &state.model
        };
        let event_type = if state.is_error { CE_TYPE_ERROR } else { CE_TYPE_USAGE };
        let mut event = json!({
            "specversion": CE_SPEC_VERSION,
            "id": event_id,
            "source": self.source,
            "type": event_type,
            "subject": state.username,
            "time": chrono::Utc::now().to_rfc3339(),
            "datacontenttype": "application/json",
            "data": {
                "user": state.username,
                "group": state.group,
                "subscription": state.subscription,
                "provider": provider,
                "model": model,
                "duration_ms": u64::try_from(state.start.elapsed().as_millis()).unwrap_or(u64::MAX),
                "user_agent": state.user_agent,
            }
        });

        if state.is_error {
            if let Some(data) = event.get_mut("data").and_then(serde_json::Value::as_object_mut) {
                data.insert("status_code".to_owned(), json!(state.response_status));
            }
        } else if let Some(data) = event.get_mut("data").and_then(serde_json::Value::as_object_mut) {
            data.insert("prompt_tokens".to_owned(), token_count(ctx, META_TOKEN_INPUT).into());
            data.insert(
                "completion_tokens".to_owned(),
                token_count(ctx, META_TOKEN_OUTPUT).into(),
            );
            data.insert("total_tokens".to_owned(), token_count(ctx, META_TOKEN_TOTAL).into());
            data.insert(
                "cached_input_tokens".to_owned(),
                token_count(ctx, META_TOKEN_CACHE_READ).into(),
            );
            data.insert(
                "cache_creation_tokens".to_owned(),
                token_count(ctx, META_TOKEN_CACHE_WRITE).into(),
            );
            data.insert(
                "reasoning_tokens".to_owned(),
                token_count(ctx, META_TOKEN_REASONING).into(),
            );
        }

        let body = match serde_json::to_vec(&event) {
            Ok(body) => body,
            Err(error) => {
                warn!(%error, "enmaas_policy: failed to serialize usage event");
                counter!("enmaas_policy_usage_report_failures_total", "reason" => "serialize").increment(1);
                return;
            },
        };
        self.spawn(body);
    }

    fn spawn(&self, body: Vec<u8>) {
        let client = self.client.clone();
        let url = self.url.clone();
        let timeout = self.timeout;
        let address_policy = self.address_policy;
        let auth_token = self.auth_token.clone();
        tokio::spawn(async move {
            let mut headers = http::HeaderMap::new();
            headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
            if let Some(token) = auth_token {
                headers.insert(AUTHORIZATION, token);
            }
            let request = SubRequest {
                method: http::Method::POST,
                uri: http::Uri::default(),
                headers,
                body: Bytes::from(body),
            };
            let result =
                subrequest::execute_configured_url(&client, &url, request, MAX_RESPONSE_BYTES, timeout, address_policy)
                    .await;
            match result {
                Ok(response) if (200..300).contains(&response.status) => {
                    trace!(status = response.status, "enmaas_policy: usage event sent");
                },
                Ok(response) => {
                    counter!("enmaas_policy_usage_report_failures_total", "reason" => "status").increment(1);
                    warn!(status = response.status, "enmaas_policy: usage event rejected");
                },
                Err(error) => {
                    counter!("enmaas_policy_usage_report_failures_total", "reason" => "transport").increment(1);
                    warn!(%error, "enmaas_policy: usage event failed");
                },
            }
        });
    }
}

fn token_count(ctx: &HttpFilterContext<'_>, key: &str) -> u64 {
    ctx.filter_metadata
        .get(key)
        .and_then(|value| value.parse().ok())
        .unwrap_or(0)
}

pub(crate) fn store_state(ctx: &mut HttpFilterContext<'_>, state: RequestState) {
    ctx.filter_state
        .insert(filter_state_key(ctx.current_filter_id), Box::new(state));
}

pub(crate) fn update_response(ctx: &mut HttpFilterContext<'_>, status: u16, is_error: bool) {
    if let Some(state) = ctx
        .filter_state
        .get_mut(&filter_state_key(ctx.current_filter_id))
        .and_then(|state| state.downcast_mut::<RequestState>())
    {
        state.response_status = status;
        state.is_error = is_error;
    }
}

pub(crate) fn take_state(ctx: &mut HttpFilterContext<'_>) -> Option<RequestState> {
    ctx.filter_state
        .remove(&filter_state_key(ctx.current_filter_id))
        .and_then(|state| state.downcast::<RequestState>().ok())
        .map(|state| *state)
}

fn filter_state_key(id: Option<usize>) -> usize {
    id.unwrap_or(usize::MAX)
}
