// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! External metering filter: pre-request balance checks and post-response
//! token usage reporting via [`CloudEvents`] to an external metering service.
//!
//! Reads token counts from [`filter_metadata`] keys set by the `token_count`
//! filter (`token.input`, `token.output`, `token.total`, and the prompt cache
//! breakdown `token.cache_read` / `token.cache_write`). The metering filter
//! must be declared *before* `token_count` in the YAML filter chain so that
//! response hooks (which run in reverse order) execute after token extraction.
//!
//! [`CloudEvents`]: https://github.com/cloudevents/spec/blob/v1.0.2/cloudevents/spec.md
//! [`filter_metadata`]: HttpFilterContext::filter_metadata

mod config;

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::needless_raw_strings,
    clippy::needless_raw_string_hashes,
    reason = "tests"
)]
mod tests;

use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use http::header::HeaderName;
use metrics::counter;
use percent_encoding::{AsciiSet, CONTROLS, utf8_percent_encode};
use praxis_ai_apis::subrequest::{self, SubRequest, SubRequestClient, SubRequestError, SubResponse};
use praxis_core::subrequest::SubRequestConnector;
use praxis_filter::{
    BodyAccess, BodyMode, FilterAction, FilterError, HttpFilter, HttpFilterContext, Rejection, parse_filter_config,
};
use serde::Deserialize;
use tracing::{debug, trace, warn};

use self::config::{ExternalMeteringConfig, validate_config};

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// `CloudEvents` spec version.
const CE_SPEC_VERSION: &str = "1.0";

/// `CloudEvent` type for provider error responses.
const CE_TYPE_ERROR: &str = "inference.request.error";

/// `CloudEvent` type for successful token usage.
const CE_TYPE_USAGE: &str = "inference.tokens.used";

/// Metadata key holding the resolved model name, written during the request
/// body phase and read during the response body phase.
const META_METERING_MODEL: &str = "metering.model";

/// Well-known `filter_metadata` key for input tokens (set by `token_count`).
const META_TOKEN_INPUT: &str = "token.input";

/// Well-known `filter_metadata` key for output tokens (set by `token_count`).
const META_TOKEN_OUTPUT: &str = "token.output";

/// Well-known `filter_metadata` key for total tokens (set by `token_count`).
const META_TOKEN_TOTAL: &str = "token.total";

/// Well-known `filter_metadata` key for prompt cache reads (set by
/// `token_count`). A breakdown of [`META_TOKEN_INPUT`], not an addition to it.
const META_TOKEN_CACHE_READ: &str = "token.cache_read";

/// Well-known `filter_metadata` key for prompt cache writes (set by
/// `token_count`). A breakdown of [`META_TOKEN_INPUT`], not an addition to it.
const META_TOKEN_CACHE_WRITE: &str = "token.cache_write";

/// Counter incremented whenever a usage or error event fails to reach the
/// metering service (transport failure or non-2xx acknowledgement).
const METRIC_REPORT_FAILURES: &str = "praxis_ai_metering_report_failures_total";

/// Upper bound on metering service response bodies.
///
/// Balance and event-acknowledgement responses are small JSON documents;
/// anything larger indicates a misbehaving service and is cut off.
const MAX_CALLOUT_RESPONSE_BYTES: usize = 64 * 1024;

/// Keepalive pool size for the private per-filter sub-request connector
/// created by [`ExternalMeteringFilter::from_config`].
const PRIVATE_POOL_SIZE: usize = 4;

/// Characters escaped when interpolating values into the balance check URL.
///
/// Deliberately narrower than [`percent_encoding::NON_ALPHANUMERIC`]: feature
/// keys and model names routinely contain hyphens and dots, which are valid
/// path characters and must survive unescaped for the metering service to
/// match them.
const PATH_SEGMENT: &AsciiSet = &CONTROLS
    .add(b' ')
    .add(b'"')
    .add(b'#')
    .add(b'%')
    .add(b'/')
    .add(b':')
    .add(b'<')
    .add(b'>')
    .add(b'?')
    .add(b'@')
    .add(b'`')
    .add(b'{')
    .add(b'}');

/// Status returned when the tenant has no remaining token budget.
const STATUS_BUDGET_EXHAUSTED: u16 = 429;

/// Status returned when metering is unreachable and `fail_open` is disabled.
const STATUS_METERING_UNAVAILABLE: u16 = 503;

// -----------------------------------------------------------------------------
// ExternalMeteringFilter
// -----------------------------------------------------------------------------

/// Integrates with an external metering service for pre-request balance
/// checks and post-response token usage reporting.
///
/// Tenant identity is resolved from the highest-trust source available:
/// verified `{prefix}*` metadata written by an authentication filter,
/// then the `identity_header_guard` filter's namespaced
/// `{namespace}.{prefix}*` metadata, then raw `{prefix}*` request
/// headers. A higher tier always wins, so forged client headers can
/// never override verified claims, and identity headers plus client
/// credentials are always stripped before the request is forwarded.
///
/// # YAML
///
/// ```yaml
/// filter: external_metering
/// metering_url: "http://metering-service:8080"
/// timeout_seconds: 5
/// feature_key: "inference-tokens"
/// source: "ai-gateway"
/// fail_open: true
/// identity_header_prefix: "x-tenant-"
/// identity_metadata_namespace: "identity"
/// default_username: "anonymous"
/// default_model: "unknown"
/// ```
pub struct ExternalMeteringFilter {
    /// Model name reported when neither the identity header nor the request
    /// body reveals one.
    default_model: Option<String>,

    /// Username reported when no identity header is present. When unset,
    /// unidentified requests are not metered at all.
    default_username: Option<String>,

    /// Whether to admit requests when the metering service is unreachable.
    fail_open: bool,

    /// Entitlement feature key used in the balance check path.
    feature_key: String,

    /// Prefix of the tenant identity headers to capture and strip.
    /// Lowercased once at construction.
    identity_header_prefix: String,

    /// Metadata namespace the `identity_header_guard` filter writes
    /// captured headers under (tier 2 of identity resolution).
    identity_metadata_namespace: String,

    /// Base URL of the external metering service.
    metering_url: String,

    /// `CloudEvents` `source` attribute for emitted events.
    source: String,

    /// Shared HTTP client for balance checks and usage reports.
    subrequest_client: SubRequestClient,

    /// HTTP timeout applied to each metering call.
    timeout: Duration,
}

impl ExternalMeteringFilter {
    /// Create from parsed YAML config with a private sub-request client.
    ///
    /// The private connector uses a small keepalive pool. Prefer
    /// [`from_config_with_client`] when a shared server-level client is
    /// available.
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] if config parsing or validation fails.
    ///
    /// [`from_config_with_client`]: Self::from_config_with_client
    pub fn from_config(config: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
        let client = SubRequestClient::new(SubRequestConnector::new(PRIVATE_POOL_SIZE, None));
        Ok(Box::new(Self::build(config, client)?))
    }

    /// Create from parsed YAML config using the shared [`SubRequestClient`].
    ///
    /// The shared client inherits the server-level pool size and
    /// connection limits from the runtime configuration.
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] if config parsing or validation fails.
    pub fn from_config_with_client(
        config: &serde_yaml::Value,
        client: SubRequestClient,
    ) -> Result<Box<dyn HttpFilter>, FilterError> {
        Ok(Box::new(Self::build(config, client)?))
    }

    /// Build the concrete filter from parsed YAML config.
    fn build(config: &serde_yaml::Value, subrequest_client: SubRequestClient) -> Result<Self, FilterError> {
        let cfg: ExternalMeteringConfig = parse_filter_config("external_metering", config)?;
        validate_config(&cfg)?;

        Ok(Self {
            default_model: cfg.default_model,
            default_username: cfg.default_username,
            fail_open: cfg.fail_open,
            feature_key: cfg.feature_key,
            identity_header_prefix: cfg.identity_header_prefix.to_ascii_lowercase(),
            identity_metadata_namespace: cfg.identity_metadata_namespace,
            metering_url: cfg.metering_url,
            source: cfg.source,
            subrequest_client,
            timeout: Duration::from_secs(cfg.timeout_seconds),
        })
    }

    /// Ask the metering service whether the tenant may spend more tokens.
    ///
    /// The metering service expresses denial only through a `2xx` response
    /// whose body carries `hasAccess: false`. Any non-`2xx` status means the
    /// service itself is misbehaving (bad route, internal error), so it is
    /// handled by the availability policy (`fail_open`) rather than treated
    /// as a denial.
    async fn check_balance(&self, state: &MeteringState) -> FilterAction {
        let url = build_balance_url(&self.metering_url, &state.username, &self.feature_key, &state.model);
        let request = SubRequest {
            method: http::Method::GET,
            uri: http::Uri::default(),
            headers: http::HeaderMap::new(),
            body: Bytes::new(),
        };

        let result = subrequest::execute_url(
            &self.subrequest_client,
            &url,
            request,
            MAX_CALLOUT_RESPONSE_BYTES,
            self.timeout,
        )
        .await;

        match result {
            Ok(resp) if is_success(resp.status) => parse_balance_result(&resp.body, self.fail_open),
            Ok(resp) => {
                warn!(status = resp.status, "balance check returned a non-2xx status");
                self.on_metering_unavailable()
            },
            Err(error) => {
                warn!(%error, "balance check unreachable");
                self.on_metering_unavailable()
            },
        }
    }

    /// Resolve the action to take when the balance check cannot be completed.
    fn on_metering_unavailable(&self) -> FilterAction {
        if self.fail_open {
            trace!("admitting request (fail-open)");
            FilterAction::Continue
        } else {
            reject_unavailable()
        }
    }

    /// Emit the terminal usage or error event for a completed request.
    fn report(&self, ctx: &HttpFilterContext<'_>, mut state: MeteringState) {
        if state.model.is_empty() {
            if let Some(model) = ctx.filter_metadata.get(META_METERING_MODEL) {
                state.model.clone_from(model);
            } else if let Some(fallback) = self.default_model.as_ref() {
                state.model.clone_from(fallback);
            }
        }

        let request_id = ctx.id_generator.generate(ctx.time_source);
        let provider = ctx.cluster_name().unwrap_or_default().to_owned();
        let event_ctx = EventContext {
            duration_ms: u64::try_from(state.request_start.elapsed().as_millis()).unwrap_or(u64::MAX),
            event_id: &request_id,
            provider: &provider,
            source: &self.source,
            state: &state,
        };

        let event = if state.is_error {
            build_error_event(&event_ctx)
        } else {
            build_usage_event(&event_ctx, &TokenCounts::read(ctx))
        };

        spawn_usage_report(self.subrequest_client.clone(), &self.metering_url, self.timeout, &event);
    }
}

#[async_trait]
impl HttpFilter for ExternalMeteringFilter {
    fn name(&self) -> &'static str {
        "external_metering"
    }

    async fn on_request(&self, ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        let mut state = capture_identity(ctx, &self.identity_header_prefix, &self.identity_metadata_namespace);

        if state.username.is_empty() {
            let Some(fallback) = self.default_username.as_ref() else {
                trace!("no tenant identity header, skipping metering");
                return Ok(FilterAction::Continue);
            };
            state.username.clone_from(fallback);
        }

        if !state.model.is_empty() {
            ctx.filter_metadata
                .insert(META_METERING_MODEL.to_owned(), state.model.clone());
        }

        let action = self.check_balance(&state).await;
        store_state(ctx, state);

        Ok(action)
    }

    fn request_body_access(&self) -> BodyAccess {
        BodyAccess::ReadOnly
    }

    fn request_body_mode(&self) -> BodyMode {
        BodyMode::Stream
    }

    async fn on_request_body(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        body: &mut Option<Bytes>,
        _end_of_stream: bool,
    ) -> Result<FilterAction, FilterError> {
        // The identity header wins when present; only fall back to the body so
        // that clients that do not send a model header are still attributed.
        let unresolved = !ctx.filter_metadata.contains_key(META_METERING_MODEL);
        if let Some(model) = body
            .as_ref()
            .filter(|_| unresolved)
            .and_then(|chunk| extract_model_from_bytes(chunk))
        {
            ctx.filter_metadata.insert(META_METERING_MODEL.to_owned(), model);
        }

        Ok(FilterAction::Release)
    }

    async fn on_response(&self, ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        let status = ctx.response_header.as_ref().map_or(0, |r| r.status.as_u16());
        let is_error = ctx.response_header.as_ref().is_some_and(|r| !r.status.is_success());

        if let Some(state) = ctx
            .filter_state
            .get_mut(&filter_state_key(ctx.current_filter_id))
            .and_then(|s| s.downcast_mut::<MeteringState>())
        {
            state.is_error = is_error;
            state.response_status = status;
        }

        Ok(FilterAction::Continue)
    }

    fn response_body_access(&self) -> BodyAccess {
        BodyAccess::ReadOnly
    }

    fn response_body_mode(&self) -> BodyMode {
        BodyMode::Stream
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

        let Some(state) = ctx
            .filter_state
            .remove(&filter_state_key(ctx.current_filter_id))
            .and_then(|s| s.downcast::<MeteringState>().ok())
        else {
            return Ok(FilterAction::Continue);
        };

        if !state.username.is_empty() {
            self.report(ctx, *state);
        }

        Ok(FilterAction::Continue)
    }
}

// -----------------------------------------------------------------------------
// Per-Request State
// -----------------------------------------------------------------------------

/// Identity and timing state captured during the request phase and consumed
/// during the response phase.
struct MeteringState {
    /// Tenant group, falling back to the subscription when absent.
    group: String,

    /// Whether the upstream returned a non-success status.
    is_error: bool,

    /// Model attributed to this request.
    model: String,

    /// Start of the request, used to derive the reported duration.
    request_start: std::time::Instant,

    /// Upstream response status, reported on error events.
    response_status: u16,

    /// Tenant subscription identifier.
    subscription: String,

    /// Client user agent, reported for attribution.
    user_agent: String,

    /// Tenant username; empty means the request is not metered.
    username: String,
}

/// Token counts published by the `token_count` filter.
struct TokenCounts {
    /// Completion tokens.
    output: u64,

    /// Prompt tokens.
    input: u64,

    /// Total tokens billed.
    total: u64,

    /// Prompt tokens served from the provider's cache; a subset of `input`.
    cache_read: u64,

    /// Prompt tokens written to the provider's cache; a subset of `input`.
    cache_write: u64,
}

impl TokenCounts {
    /// Read the counts the `token_count` filter left in `filter_metadata`.
    fn read(ctx: &HttpFilterContext<'_>) -> Self {
        Self {
            input: read_token_meta(ctx, META_TOKEN_INPUT),
            output: read_token_meta(ctx, META_TOKEN_OUTPUT),
            total: read_token_meta(ctx, META_TOKEN_TOTAL),
            cache_read: read_token_meta(ctx, META_TOKEN_CACHE_READ),
            cache_write: read_token_meta(ctx, META_TOKEN_CACHE_WRITE),
        }
    }
}

/// Fields shared by the usage and error `CloudEvents`.
struct EventContext<'a> {
    /// Wall-clock duration of the proxied request.
    duration_ms: u64,

    /// Unique `CloudEvents` `id`.
    event_id: &'a str,

    /// Upstream cluster the request was routed to.
    provider: &'a str,

    /// Configured `CloudEvents` `source`.
    source: &'a str,

    /// Captured tenant identity.
    state: &'a MeteringState,
}

/// JSON response from the metering service balance check endpoint.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BalanceResponse {
    /// Whether the tenant may spend more tokens.
    has_access: bool,
}

// -----------------------------------------------------------------------------
// Identity Header Capture
// -----------------------------------------------------------------------------

/// Capture tenant identity from the configured headers and mark those headers,
/// along with client credentials, for removal before the request is forwarded.
///
/// `prefix` must already be lowercase; [`ExternalMeteringFilter::build`]
/// lowercases it once at construction.
fn capture_identity(ctx: &mut HttpFilterContext<'_>, prefix: &str, identity_namespace: &str) -> MeteringState {
    let user_agent = ctx
        .request
        .headers
        .get("user-agent")
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .to_owned();

    let mut identity = read_identity_headers(ctx, prefix, identity_namespace);
    strip_client_credentials(ctx);

    if identity.group.is_empty() {
        identity.group.clone_from(&identity.subscription);
    }

    MeteringState {
        group: identity.group,
        is_error: false,
        model: identity.model,
        request_start: std::time::Instant::now(),
        response_status: 0,
        subscription: identity.subscription,
        user_agent,
        username: identity.username,
    }
}

/// Tenant identity as carried on the request headers.
#[derive(Default)]
struct Identity {
    /// Value of `{prefix}group`.
    group: String,

    /// Value of `{prefix}model`.
    model: String,

    /// Value of `{prefix}subscription`.
    subscription: String,

    /// Value of `{prefix}username`.
    username: String,
}

/// Resolve tenant identity from the highest-trust source available.
///
/// Three tiers, most trusted first:
///
/// 1. Unnamespaced `{prefix}*` metadata keys, written by an authentication filter from verified credentials (e.g. JWT
///    claims or a validated API key). When any of these are present, every lower tier is ignored entirely so a client
///    cannot spoof the remaining fields via forged headers alongside valid credentials.
/// 2. Namespaced `{namespace}.{prefix}*` metadata keys, written by the `identity_header_guard` filter from captured
///    request headers.
/// 3. Raw `{prefix}*` request headers, set by a trusted upstream auth layer when neither metadata tier is populated.
///
/// Identity headers are always marked for removal so tenant identity
/// never leaks to the upstream provider, regardless of which tier
/// supplied the identity.
fn read_identity_headers(ctx: &mut HttpFilterContext<'_>, prefix: &str, identity_namespace: &str) -> Identity {
    let mut identity = Identity::default();
    read_metadata_identity(ctx, prefix, "", &mut identity);

    // Any verified field blocks the lower tiers entirely: an auth filter
    // may map only some claims (e.g. group without username), and a
    // partially verified identity must not be extended by forgeable
    // sources.
    let has_verified_identity = !(identity.username.is_empty()
        && identity.group.is_empty()
        && identity.subscription.is_empty()
        && identity.model.is_empty());

    if !has_verified_identity {
        read_metadata_identity(ctx, prefix, &format!("{identity_namespace}."), &mut identity);
    }

    let has_metadata_identity = has_verified_identity || !identity.username.is_empty();
    if has_metadata_identity {
        strip_identity_headers(ctx, prefix);
    } else {
        read_header_identity(ctx, prefix, &mut identity);
    }

    identity
}

/// Read the `{namespace}{prefix}*` identity keys from `filter_metadata`,
/// overwriting only the fields the namespace carries.
fn read_metadata_identity(ctx: &HttpFilterContext<'_>, prefix_lower: &str, namespace: &str, identity: &mut Identity) {
    if let Some(val) = ctx.filter_metadata.get(&format!("{namespace}{prefix_lower}group")) {
        identity.group.clone_from(val);
    }
    if let Some(val) = ctx.filter_metadata.get(&format!("{namespace}{prefix_lower}model")) {
        identity.model.clone_from(val);
    }
    if let Some(val) = ctx
        .filter_metadata
        .get(&format!("{namespace}{prefix_lower}subscription"))
    {
        identity.subscription.clone_from(val);
    }
    if let Some(val) = ctx.filter_metadata.get(&format!("{namespace}{prefix_lower}username")) {
        identity.username.clone_from(val);
    }
}

/// Read identity from raw `{prefix}*` request headers, marking each one
/// for removal.
///
/// [`HeaderName::as_str`] already returns the lowercased name, so the
/// pre-lowercased prefix compares directly without per-header allocation.
fn read_header_identity(ctx: &mut HttpFilterContext<'_>, prefix_lower: &str, identity: &mut Identity) {
    for (key, value) in &ctx.request.headers {
        let Some(suffix) = key.as_str().strip_prefix(prefix_lower) else {
            continue;
        };
        let val = value.to_str().unwrap_or_default();

        match suffix {
            "group" if identity.group.is_empty() => val.clone_into(&mut identity.group),
            "model" if identity.model.is_empty() => val.clone_into(&mut identity.model),
            "subscription" if identity.subscription.is_empty() => val.clone_into(&mut identity.subscription),
            "username" if identity.username.is_empty() => val.clone_into(&mut identity.username),
            _ => {},
        }

        ctx.request_headers_to_remove.push(key.clone());
    }
}

/// Mark every `{prefix}*` header for removal without reading it, so
/// unused identity headers still never reach the upstream provider.
fn strip_identity_headers(ctx: &mut HttpFilterContext<'_>, prefix_lower: &str) {
    for (key, _) in &ctx.request.headers {
        if key.as_str().starts_with(prefix_lower) {
            ctx.request_headers_to_remove.push(key.clone());
        }
    }
}

/// Mark client-supplied credentials for removal.
///
/// `accept-encoding` is stripped alongside them so the upstream returns an
/// uncompressed body: `token_count` parses the response inline and cannot read
/// usage out of a compressed stream.
fn strip_client_credentials(ctx: &mut HttpFilterContext<'_>) {
    ctx.request_headers_to_remove.push(http::header::AUTHORIZATION);
    ctx.request_headers_to_remove.push(http::header::ACCEPT_ENCODING);
    ctx.request_headers_to_remove.push(HeaderName::from_static("x-api-key"));
}

// -----------------------------------------------------------------------------
// Balance Check
// -----------------------------------------------------------------------------

/// Interpret a balance check response body.
fn parse_balance_result(body: &[u8], fail_open: bool) -> FilterAction {
    let Some(balance) = decode_balance(body) else {
        return admit_or_reject(fail_open);
    };

    if balance.has_access {
        trace!("balance check passed");
        FilterAction::Continue
    } else {
        reject_budget_exhausted()
    }
}

/// Decode a balance payload, logging why an unusable one was discarded.
fn decode_balance(body: &[u8]) -> Option<BalanceResponse> {
    if body.is_empty() {
        debug!("balance check returned an empty body");
        return None;
    }

    match serde_json::from_slice::<BalanceResponse>(body) {
        Ok(balance) => Some(balance),
        Err(e) => {
            debug!("balance response parse error: {e}");
            None
        },
    }
}

/// Reject with a 429 because the tenant has no remaining token budget.
fn reject_budget_exhausted() -> FilterAction {
    debug!("token budget exhausted");
    FilterAction::Reject(
        Rejection::status(STATUS_BUDGET_EXHAUSTED).with_body(Bytes::from_static(b"token budget exhausted")),
    )
}

/// Admit the request when configured to fail open, reject it otherwise.
fn admit_or_reject(fail_open: bool) -> FilterAction {
    if fail_open {
        FilterAction::Continue
    } else {
        reject_unavailable()
    }
}

/// Reject with a 503 because metering could not authorize the request.
fn reject_unavailable() -> FilterAction {
    FilterAction::Reject(
        Rejection::status(STATUS_METERING_UNAVAILABLE).with_body(Bytes::from_static(b"metering system unavailable")),
    )
}

// -----------------------------------------------------------------------------
// Usage Reporting (fire-and-forget)
// -----------------------------------------------------------------------------

/// Deliver an event without blocking the response.
///
/// Metering is an observer: a slow or failing metering service must never add
/// latency to, or fail, a request the upstream already answered.
fn spawn_usage_report(client: SubRequestClient, metering_url: &str, timeout: Duration, event: &serde_json::Value) {
    let url = format!("{}/api/v1/events", metering_url.trim_end_matches('/'));

    let body = match serde_json::to_vec(event) {
        Ok(b) => b,
        Err(e) => {
            warn!("failed to serialize metering event: {e}");
            return;
        },
    };

    tokio::spawn(async move {
        let mut headers = http::HeaderMap::new();
        headers.insert(
            http::header::CONTENT_TYPE,
            http::HeaderValue::from_static("application/json"),
        );
        let request = SubRequest {
            method: http::Method::POST,
            uri: http::Uri::default(),
            headers,
            body: Bytes::from(body),
        };

        report_delivery(subrequest::execute_url(&client, &url, request, MAX_CALLOUT_RESPONSE_BYTES, timeout).await);
    });
}

/// Log the outcome of a usage report delivery and count failures.
///
/// Emits [`METRIC_REPORT_FAILURES`] so dropped billing events are visible
/// on a dashboard rather than only in debug logs.
fn report_delivery(result: Result<SubResponse, SubRequestError>) {
    match result {
        Ok(resp) if is_success(resp.status) => {
            trace!(status = resp.status, "metering usage report sent");
        },
        Ok(resp) => {
            counter!(METRIC_REPORT_FAILURES).increment(1);
            warn!(status = resp.status, "metering usage report rejected");
        },
        Err(error) => {
            counter!(METRIC_REPORT_FAILURES).increment(1);
            warn!(%error, "metering usage report failed");
        },
    }
}

/// Whether an HTTP status code is in the 2xx range.
fn is_success(status: u16) -> bool {
    (200..300).contains(&status)
}

// -----------------------------------------------------------------------------
// URL Construction
// -----------------------------------------------------------------------------

/// Build the entitlement balance check URL for a tenant and model.
fn build_balance_url(base_url: &str, customer_id: &str, feature_key: &str, model: &str) -> String {
    let base = base_url.trim_end_matches('/');
    let customer = utf8_percent_encode(customer_id, PATH_SEGMENT);
    let feature = utf8_percent_encode(feature_key, PATH_SEGMENT);
    let model = utf8_percent_encode(model, PATH_SEGMENT);

    format!("{base}/api/v1/customers/{customer}/entitlements/{feature}/value?model={model}")
}

// -----------------------------------------------------------------------------
// CloudEvent Construction
// -----------------------------------------------------------------------------

/// Build an `inference.tokens.used` event.
fn build_usage_event(ctx: &EventContext<'_>, tokens: &TokenCounts) -> serde_json::Value {
    let mut event = build_envelope(ctx, CE_TYPE_USAGE);

    if let Some(data) = event.get_mut("data").and_then(serde_json::Value::as_object_mut) {
        data.insert("prompt_tokens".to_owned(), tokens.input.into());
        data.insert("completion_tokens".to_owned(), tokens.output.into());
        data.insert("total_tokens".to_owned(), tokens.total.into());
        data.insert("cached_input_tokens".to_owned(), tokens.cache_read.into());
        data.insert("cache_creation_tokens".to_owned(), tokens.cache_write.into());
    }

    event
}

/// Build an `inference.request.error` event.
fn build_error_event(ctx: &EventContext<'_>) -> serde_json::Value {
    let mut event = build_envelope(ctx, CE_TYPE_ERROR);

    if let Some(data) = event.get_mut("data").and_then(serde_json::Value::as_object_mut) {
        data.insert("status_code".to_owned(), ctx.state.response_status.into());
    }

    event
}

/// Build the `CloudEvents` envelope and the attribution fields both event
/// types carry.
fn build_envelope(ctx: &EventContext<'_>, event_type: &str) -> serde_json::Value {
    let state = ctx.state;

    serde_json::json!({
        "specversion": CE_SPEC_VERSION,
        "id": ctx.event_id,
        "source": ctx.source,
        "type": event_type,
        "subject": state.username,
        "time": chrono::Utc::now().to_rfc3339(),
        "datacontenttype": "application/json",
        "data": {
            "user": state.username,
            "group": state.group,
            "subscription": state.subscription,
            "provider": ctx.provider,
            "model": state.model,
            "duration_ms": ctx.duration_ms,
            "user_agent": state.user_agent,
        }
    })
}

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

/// Extract the top-level `model` field from a JSON body fragment.
///
/// Request bodies arrive in chunks and can be megabytes long, so this scans
/// with a string- and depth-aware state machine instead of buffering and
/// deserializing the whole document. Only a `"model"` key belonging to the
/// top-level object is matched, so `"model"` occurrences nested inside
/// `messages` content can never misattribute the request. Returns `None`
/// when the chunk does not contain the complete top-level `"model": "..."`
/// pair.
fn extract_model_from_bytes(bytes: &[u8]) -> Option<String> {
    let text = std::str::from_utf8(bytes).ok()?;
    let mut scanner = TopLevelKeyScanner::new(text);

    while let Some(key) = scanner.next_top_level_key() {
        if key == "model" {
            return scanner.string_value().map(str::to_owned);
        }
    }

    None
}

/// Incremental scanner over the top-level keys of a JSON object fragment.
///
/// Tracks nesting depth and string boundaries (including escapes) so keys
/// inside nested objects, arrays, or string values are never surfaced.
struct TopLevelKeyScanner<'a> {
    /// Remaining unscanned input.
    rest: &'a str,

    /// Current object/array nesting depth; the document object is depth 1.
    depth: u32,
}

impl<'a> TopLevelKeyScanner<'a> {
    /// Start scanning at the beginning of a JSON document fragment.
    fn new(text: &'a str) -> Self {
        Self { rest: text, depth: 0 }
    }

    /// Advance to the next key of the top-level object and return it.
    fn next_top_level_key(&mut self) -> Option<&'a str> {
        loop {
            let mut chars = self.rest.char_indices();
            let (pos, ch) = chars.next()?;
            match ch {
                '{' | '[' => {
                    self.depth = self.depth.checked_add(1)?;
                    self.rest = self.rest.get(pos + 1..)?;
                },
                '}' | ']' => {
                    self.depth = self.depth.checked_sub(1)?;
                    self.rest = self.rest.get(pos + 1..)?;
                },
                '"' => {
                    let start = pos + 1;
                    let end = find_string_end(self.rest, start)?;
                    let content = self.rest.get(start..end)?;
                    let after = self.rest.get(end + 1..)?;
                    let is_key = after.trim_start().starts_with(':');
                    self.rest = after;
                    if self.depth == 1 && is_key {
                        return Some(content);
                    }
                },
                _ => {
                    self.rest = self.rest.get(pos + ch.len_utf8()..)?;
                },
            }
        }
    }

    /// Read the string value following the key just returned.
    ///
    /// Returns `None` when the value is not a string (e.g. `null`) or the
    /// fragment is cut off before the closing quote.
    fn string_value(&self) -> Option<&'a str> {
        let after_colon = self.rest.trim_start().strip_prefix(':')?;
        let value = after_colon.trim_start().strip_prefix('"')?;
        let end = find_string_end(value, 0)?;
        value.get(..end)
    }
}

/// Find the byte offset of the unescaped closing quote for the string
/// starting at `from` (which must point just past the opening quote).
fn find_string_end(text: &str, from: usize) -> Option<usize> {
    let mut escaped = false;
    for (offset, ch) in text.get(from..)?.char_indices() {
        if escaped {
            escaped = false;
        } else if ch == '\\' {
            escaped = true;
        } else if ch == '"' {
            return Some(from + offset);
        }
    }
    None
}

/// Read a numeric `filter_metadata` value, defaulting to zero.
fn read_token_meta(ctx: &HttpFilterContext<'_>, key: &str) -> u64 {
    ctx.filter_metadata.get(key).and_then(|v| v.parse().ok()).unwrap_or(0)
}

/// Key under which this filter instance stores its per-request state.
fn filter_state_key(id: Option<usize>) -> usize {
    id.unwrap_or(0)
}

/// Store per-request state for retrieval during the response phase.
fn store_state(ctx: &mut HttpFilterContext<'_>, state: MeteringState) {
    let key = filter_state_key(ctx.current_filter_id);
    ctx.filter_state.insert(key, Box::new(state));
}
