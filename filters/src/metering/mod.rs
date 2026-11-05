// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! External metering filter: pre-request balance checks and post-response
//! token usage reporting via [`CloudEvents`] to an external metering service.
//!
//! Reads token counts from [`filter_metadata`] keys set by the `token_count`
//! filter (`token.input`, `token.output`, `token.total`, the prompt cache
//! breakdown `token.cache_read` / `token.cache_write`, and the reasoning
//! breakdown `token.reasoning`). The metering filter
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
    clippy::panic,
    reason = "tests"
)]
mod tests;

use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use http::header::HeaderName;
use metrics::counter;
use percent_encoding::{AsciiSet, CONTROLS, utf8_percent_encode};
use praxis_ai_apis::{
    MODEL_PROVIDER_CLIENT_MODEL_METADATA,
    callout_target::AddressPolicy,
    credential_source::{self, CredentialSource, DiagnosticContext, SecretHandle, WatchHandle},
    subrequest::{self, SubRequest, SubRequestClient, SubRequestError, SubResponse},
};
use praxis_filter::{
    BodyAccess, BodyMode, FilterAction, FilterError, HttpFilter, HttpFilterContext, Rejection,
    body::DEFAULT_JSON_BODY_MAX_BYTES, parse_filter_config,
};
use tracing::{debug, trace, warn};

use self::config::{ExternalMeteringConfig, validate_config};
// Import the token metadata keys from `token_count`, their sole writer, so this
// reader can never drift to a stale literal and silently meter zero tokens.
use crate::token_usage::{
    META_TOKEN_CACHE_READ, META_TOKEN_CACHE_WRITE, META_TOKEN_INPUT, META_TOKEN_OUTPUT, META_TOKEN_REASONING,
    META_TOKEN_TOTAL,
};

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
pub(crate) const META_METERING_MODEL: &str = "metering.model";

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

/// Status returned when the model-policy gate explicitly denies the
/// requested model.
const STATUS_MODEL_NOT_ALLOWED: u16 = 403;

/// Status returned when model policy is enabled but authorization cannot
/// be evaluated: no trusted tenant identity or no resolvable public model.
const STATUS_AUTHZ_INPUT_REQUIRED: u16 = 403;

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
/// Both event types carry `data.organization_id` and `data.cost_center`
/// from `{prefix}organization-id` and `{prefix}cost-center`. When absent,
/// those fields are empty strings rather than invented defaults.
///
/// When no `{prefix}model` identity header names the model and
/// `model_to_provider` mapped the request, balance checks and usage events use
/// the client-facing model ID it recorded rather than the provider target.
/// That filter buffers the request body, so its body pass completes before
/// this filter's request hook runs the balance check.
///
/// Ordering: declare `external_metering` *before* `token_count` in the filter
/// chain so response hooks (which run in reverse order) execute after token
/// extraction, and after any verification filter (`identity_header_guard`,
/// authentication) so captured identity metadata is already present. The
/// model-policy preflight always runs before provider dispatch: after a
/// rejection, zero provider requests have been made.
///
/// Response contract (stable bodies): an explicit budget denial rejects with
/// 429 and body `token budget exhausted`; an explicit model denial rejects
/// with 403 and body `model not allowed for this user`, regardless of either
/// `fail_open` setting and with precedence over budget exhaustion; a verdict
/// unavailable under a fail-closed gate rejects with 503 and body
/// `metering system unavailable`. With model policy enabled, a missing trusted
/// identity rejects with 403 and body `tenant identity required for model
/// authorization`; an unresolvable model rejects with 403 and body `request
/// model required for model authorization`; conflicting or ambiguous model
/// sources reject with 403 and body `conflicting model inputs for model
/// authorization`; and a
/// request body larger than the pre-read bound rejects with 413 before
/// dispatch.
///
/// Buffering: with model policy enabled the request body is pre-read once,
/// bounded by the standard JSON body maximum (10 MiB); requests exceeding the
/// bound reject with 413. Only the top-level `"model"` field is inspected —
/// the body is never re-serialized or copied for the preflight. Once
/// `model_to_provider` recorded a client-facing model, the buffered body has
/// already been rewritten to the *provider* target, so it is no longer a
/// model source: the preflight then compares the identity claim against the
/// recorded client-facing model alone.
///
/// # Model policy (optional)
///
/// With `model_policy_check: true`, the same entitlement preflight also
/// enforces the metering service's model-policy verdict (`modelAllowed`):
///
/// - `modelAllowed: false` rejects with **403** before any provider request, regardless of the billing `fail_open` and
///   with precedence over budget exhaustion (`hasAccess: false` still rejects with 429 on its own).
/// - An unavailable model verdict — missing, `null`, wrongly typed, an unusable payload, a non-2xx response, a timeout,
///   or a transport error — is governed by `model_policy_fail_open` (default **fail-closed, 503**), independent of the
///   billing `fail_open`. Admission on a shared callout failure requires every enabled gate's availability policy to
///   permit it.
/// - Every metered request then needs a trusted tenant identity and a resolvable public model. If the identity comes
///   from raw `{prefix}*` request headers, the ingress boundary must delete client-supplied copies and set those
///   headers from authenticated data before this filter runs: the filter cannot distinguish client-supplied headers
///   from trusted ones, so a client-set header would choose the identity used for model authorization. Every available
///   model source — the identity header, the `model_to_provider` client model, and (when no client model was recorded)
///   the request body's top-level `"model"` (bounded pre-read) — must agree; conflicting non-empty sources, or a body
///   `"model"` member that is ambiguous (duplicate or escaped key spellings) or undecodable (escaped or non-string
///   value), reject with 403 before dispatch, since authorizing one source would not authorize the model the provider
///   receives. Reporting fallbacks (`default_username`, `default_model`) and the skip-metering path never bypass
///   authorization: such requests reject with 403 before dispatch.
///
/// With `model_policy_check: false` (default), the legacy contract is
/// preserved exactly: only `hasAccess` decides, budget denial is 429, and
/// the billing `fail_open` governs availability.
///
/// # YAML
///
/// ```yaml
/// filter: external_metering
/// metering_url: "http://metering-service:8080"
/// timeout_seconds: 5
/// feature_key: "inference-tokens"
/// source: "ai-gateway"
/// provider: "openai"
/// fail_open: true
/// identity_header_prefix: "x-tenant-"
/// identity_metadata_namespace: "identity"
/// default_username: "anonymous"
/// default_model: "unknown"
/// model_policy_check: true
/// model_policy_fail_open: false
/// ```
#[expect(
    clippy::struct_excessive_bools,
    reason = "independent billing and model-policy fail-open controls"
)]
pub struct ExternalMeteringFilter {
    /// Connect-time policy for the metering endpoint. Rejects non-public
    /// addresses unless the operator sets `allow_private_endpoint`.
    address_policy: AddressPolicy,

    /// Published handle to the internal metering credential, or `None` when no
    /// `internal_auth_token` is configured (callouts stay unauthenticated). The
    /// handle's payload is the pre-assembled `Authorization` header value; a
    /// `None` payload (post-startup rotation gap) fails the callout closed.
    auth_token: Option<SecretHandle>,

    /// Retains the file watcher (when the token is `file`-sourced) for the
    /// filter's lifetime so rotation keeps publishing; dropping it stops the
    /// watcher. `None` for `value`/`env_var` sources and when no token is set.
    _auth_watch: Option<WatchHandle>,

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

    /// Whether the entitlement preflight enforces the model-policy verdict
    /// (`modelAllowed`). See [`ExternalMeteringConfig::model_policy_check`].
    model_policy_check: bool,

    /// Availability policy for the model-policy gate, independent of the
    /// billing `fail_open`. See
    /// [`ExternalMeteringConfig::model_policy_fail_open`].
    model_policy_fail_open: bool,

    /// Static provider name for emitted events. Takes precedence over the
    /// routed cluster name; the only source of provider attribution on the
    /// ext-proc data path, where `ctx.cluster_name()` is always `None`.
    provider: Option<String>,

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
        let client = crate::isolated_subrequest_client(PRIVATE_POOL_SIZE);
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

        let (auth_token, auth_watch) = build_auth_token(cfg.internal_auth_token.as_ref())?;

        Ok(Self {
            address_policy: AddressPolicy::from_allow_private(cfg.allow_private_endpoint),
            auth_token,
            _auth_watch: auth_watch,
            default_model: cfg.default_model,
            default_username: cfg.default_username,
            fail_open: cfg.fail_open,
            feature_key: cfg.feature_key,
            identity_header_prefix: cfg.identity_header_prefix.to_ascii_lowercase(),
            identity_metadata_namespace: cfg.identity_metadata_namespace,
            metering_url: cfg.metering_url,
            model_policy_check: cfg.model_policy_check,
            model_policy_fail_open: cfg.model_policy_fail_open,
            provider: cfg.provider,
            source: cfg.source,
            subrequest_client,
            timeout: Duration::from_secs(cfg.timeout_seconds),
        })
    }

    /// Ask the metering service whether the tenant may spend more tokens
    /// and, when model policy is enabled, whether the tenant may use the
    /// requested model.
    ///
    /// The metering service expresses denial only through a `2xx` response
    /// whose body carries `hasAccess: false` (budget) or `modelAllowed:
    /// false` (model policy). Any non-`2xx` status means the service itself
    /// is misbehaving (bad route, internal error), so it is handled by each
    /// gate's availability policy rather than treated as a denial.
    async fn check_balance(&self, state: &MeteringState) -> FilterAction {
        let url = build_balance_url(&self.metering_url, &state.username, &self.feature_key, &state.model);
        // Resolve the internal-auth header (fail closed if a configured token
        // is unavailable) before dialing the metering service.
        let headers = match self.balance_callout_headers(self.auth_header()) {
            Ok(headers) => headers,
            Err(action) => return action,
        };

        // The metering service URL is operator-configured, so it resolves
        // through the shared DNS cache.
        let result = self
            .subrequest_client
            .execute_url(
                &url,
                balance_probe_request(headers),
                subrequest::address_policy_hook(self.address_policy),
                subrequest::UrlResolutionPolicy::OperatorCached,
                self.timeout,
                MAX_CALLOUT_RESPONSE_BYTES,
                None,
            )
            .await
            .map_err(subrequest::map_url_error);

        self.interpret_balance(result)
    }

    /// Interpret the balance callout transport result into a filter action.
    ///
    /// A `2xx` response carries real per-gate verdicts; any other status or a
    /// transport error means the service is misbehaving, so every gate's
    /// availability policy applies.
    fn interpret_balance(&self, result: Result<SubResponse, SubRequestError>) -> FilterAction {
        match result {
            Ok(resp) if is_success(resp.status) => parse_balance_result(&resp.body, self.admission_policy()),
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

    /// Resolve the action to take when the balance check cannot be completed:
    /// no gate has a verdict, so admission requires every enabled gate's
    /// availability policy to permit it.
    fn on_metering_unavailable(&self) -> FilterAction {
        decide_admission(&BalanceVerdicts::unavailable(), self.admission_policy())
    }

    /// Per-gate admission settings resolved from this filter's config.
    fn admission_policy(&self) -> AdmissionPolicy {
        AdmissionPolicy {
            billing_fail_open: self.fail_open,
            model_policy_check: self.model_policy_check,
            model_policy_fail_open: self.model_policy_fail_open,
        }
    }

    /// Resolve the internal-auth state for an outgoing metering callout from the
    /// live credential handle.
    fn auth_header(&self) -> AuthOutcome {
        let Some(handle) = self.auth_token.as_ref() else {
            return AuthOutcome::Unauthenticated;
        };
        let snapshot = handle.current();
        auth_outcome_from_payload(snapshot.payload.as_deref().map(String::as_str))
    }

    /// Resolve the balance callout headers, or the fail-closed action when a
    /// configured token is currently unavailable — never dial unauthenticated.
    /// The token gap is treated as a callout failure, so each gate's
    /// availability policy decides.
    fn balance_callout_headers(&self, outcome: AuthOutcome) -> Result<http::HeaderMap, FilterAction> {
        authed_headers(outcome).ok_or_else(|| {
            warn!("internal metering auth token unavailable; balance check failing closed");
            self.on_metering_unavailable()
        })
    }

    /// Resolve the model to report: the captured model, else the
    /// body/header metadata, else the configured default.
    fn resolve_report_model(&self, ctx: &HttpFilterContext<'_>, state: &MeteringState) -> String {
        if !state.model.is_empty() {
            return state.model.clone();
        }
        if let Some(model) = ctx.get_metadata(MODEL_PROVIDER_CLIENT_MODEL_METADATA) {
            return model.to_owned();
        }
        if let Some(model) = ctx.filter_metadata.get(META_METERING_MODEL) {
            return model.clone();
        }
        self.default_model.clone().unwrap_or_default()
    }

    /// Resolve the provider to attribute usage to: the static config value,
    /// else the routed cluster name, else empty.
    fn resolve_provider<'a>(&'a self, ctx: &'a HttpFilterContext<'_>) -> &'a str {
        self.provider
            .as_deref()
            .or_else(|| ctx.cluster_name())
            .unwrap_or_default()
    }

    /// Emit the terminal usage or error event for a completed request.
    fn report(&self, ctx: &HttpFilterContext<'_>, mut state: MeteringState) {
        state.model = self.resolve_report_model(ctx, &state);

        let request_id = ctx.id_generator.generate(ctx.time_source);
        let event_ctx = EventContext {
            duration_ms: u64::try_from(state.request_start.elapsed().as_millis()).unwrap_or(u64::MAX),
            event_id: &request_id,
            provider: self.resolve_provider(ctx),
            source: &self.source,
            state: &state,
        };

        let event = if state.is_error {
            build_error_event(&event_ctx)
        } else {
            build_usage_event(&event_ctx, &TokenCounts::read(ctx))
        };

        // Resolve the callout credential on the filter (not in the spawned
        // task) so a rotation gap fails the report closed rather than sending
        // it unauthenticated.
        let Some(headers) = report_callout_headers(self.auth_header()) else {
            return;
        };

        spawn_usage_report(
            self.subrequest_client.clone(),
            &self.metering_url,
            self.timeout,
            self.address_policy,
            headers,
            &event,
        );
    }
}

#[async_trait]
impl HttpFilter for ExternalMeteringFilter {
    fn name(&self) -> &'static str {
        "external_metering"
    }

    async fn on_request(&self, ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        let mut state = capture_identity(ctx, &self.identity_header_prefix, &self.identity_metadata_namespace);
        if resolve_request_model(ctx, &mut state, self.model_policy_check) == ModelResolution::Conflict {
            return Ok(reject_model_conflict());
        }

        if state.username.is_empty() {
            if self.model_policy_check {
                // Authorization cannot be evaluated without a trusted tenant
                // identity, and reporting fallbacks must not stand in for
                // one — reject before dispatch rather than skip.
                return Ok(reject_identity_required());
            }
            let Some(fallback) = self.default_username.as_ref() else {
                trace!("no tenant identity header, skipping metering");
                return Ok(FilterAction::Continue);
            };
            state.username.clone_from(fallback);
        }

        if self.model_policy_check && state.model.is_empty() {
            // `default_model` is a reporting fallback only; authorizing it
            // would not authorize the model actually being requested.
            return Ok(reject_model_required());
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
        // Model-policy enforcement must decide before provider dispatch, so
        // the public model has to be known during `on_request`. The bounded
        // pre-read makes the body available there; requests exceeding the
        // bound reject with 413. Pipelines whose body mode already buffers
        // (e.g. because of `model_to_provider`) need no extra buffering.
        if self.model_policy_check {
            BodyMode::StreamBuffer {
                max_bytes: Some(DEFAULT_JSON_BODY_MAX_BYTES),
            }
        } else {
            BodyMode::Stream
        }
    }

    async fn on_request_body(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        body: &mut Option<Bytes>,
        _end_of_stream: bool,
    ) -> Result<FilterAction, FilterError> {
        // The identity header wins when present; only fall back to the body so
        // that clients that do not send a model header are still attributed.
        // An unusable body model contributes nothing: with enforcement
        // enabled it already rejected during `on_request`, and without it the
        // attribution must not guess either.
        let unresolved = !ctx.filter_metadata.contains_key(META_METERING_MODEL);
        if unresolved && let Some(BodyModel::Usable(model)) = body.as_ref().map(|chunk| extract_model_from_bytes(chunk))
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
    /// Cost center allocated for billing.
    cost_center: String,

    /// Tenant group, falling back to the subscription when absent.
    group: String,

    /// Whether the upstream returned a non-success status.
    is_error: bool,

    /// Model attributed to this request.
    model: String,

    /// Billing organization identifier.
    organization_id: String,

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

    /// Reasoning/thinking tokens; a subset of `output`.
    reasoning: u64,
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
            reasoning: read_token_meta(ctx, META_TOKEN_REASONING),
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

/// JSON response from the metering service balance check endpoint, parsed
/// per gate so one unusable verdict never discards the other.
///
/// `None` means the gate has no verdict: the field was missing, `null`, or
/// wrongly typed, or the whole payload was unusable. An unavailable verdict
/// is never an implicit allow and never an explicit denial.
struct BalanceVerdicts {
    /// Whether the tenant may spend more tokens.
    has_access: Option<bool>,

    /// Whether the tenant may use the requested model.
    model_allowed: Option<bool>,
}

impl BalanceVerdicts {
    /// Verdicts for a callout that produced no usable response at all
    /// (transport error, non-2xx status, empty or malformed body).
    fn unavailable() -> Self {
        Self {
            has_access: None,
            model_allowed: None,
        }
    }
}

/// Per-gate admission settings resolved from the filter config.
///
/// The billing gate is always enabled; the model-policy gate is opt-in and
/// carries its own availability policy, so a billing `fail_open` can never
/// silently drag the model gate open (or closed).
#[derive(Clone, Copy)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "per-gate enable and availability controls that must stay independent"
)]
struct AdmissionPolicy {
    /// Billing-gate availability: admit when the budget verdict is
    /// unavailable (the legacy `fail_open` setting).
    billing_fail_open: bool,

    /// Whether the model-policy verdict (`modelAllowed`) is enforced.
    model_policy_check: bool,

    /// Model-gate availability: admit when the model verdict is
    /// unavailable. Independent of `billing_fail_open`.
    model_policy_fail_open: bool,
}

// -----------------------------------------------------------------------------
// Identity Header Capture
// -----------------------------------------------------------------------------

/// Outcome of resolving the public model for the entitlement preflight.
#[derive(PartialEq)]
enum ModelResolution {
    /// Resolution succeeded, or no model is available (the caller applies the
    /// missing-model rejection when enforcement needs one).
    Continue,
    /// Non-empty model sources disagree: authorizing either one would admit a
    /// request whose provider-bound body names a different model.
    Conflict,
}

/// Resolve the public model for the entitlement preflight.
///
/// Sources: the trusted `{prefix}model` identity header (captured by
/// [`capture_identity`]), the `model_to_provider` client-facing model (the
/// public alias, recorded before any provider rewrite), and — only when
/// model-policy enforcement needs the model before dispatch and
/// `model_to_provider` recorded no client-facing model — the request body's
/// top-level `"model"` field from the bounded pre-read buffer. The body is
/// skipped once a client-facing model exists because that filter's rewrite
/// has already replaced it with the *provider* target, which legitimately
/// differs from the public alias. With enforcement enabled every available
/// source must name the same model: a conflict — or a body `"model"` member
/// that is ambiguous (duplicate or escaped key spellings) or undecodable
/// (escaped or non-string value), which the provider's parser may resolve
/// differently — rejects rather than authorizing a model the request does not
/// actually use. With enforcement
/// disabled the legacy reporting contract is preserved: the identity model,
/// then the client-facing model, and the body is consulted later for
/// attribution only. `default_model` is deliberately not consulted: it is a
/// reporting fallback, never an authorization input.
fn resolve_request_model(
    ctx: &HttpFilterContext<'_>,
    state: &mut MeteringState,
    model_policy_check: bool,
) -> ModelResolution {
    let client_model = ctx
        .get_metadata(MODEL_PROVIDER_CLIENT_MODEL_METADATA)
        .filter(|model| !model.is_empty());
    let body_model = match ctx.buffered_request_body.as_ref() {
        Some(body) if client_model.is_none() && model_policy_check => extract_model_from_bytes(body),
        _ => BodyModel::Absent,
    };

    if model_policy_check {
        // An unusable body model must not be treated as absent: the
        // provider's parser resolves the ambiguity its own way, so the
        // request may carry a model the preflight never compared against
        // the identity or client claim.
        if matches!(body_model, BodyModel::Unusable) {
            return ModelResolution::Conflict;
        }
        if model_sources_conflict(&state.model, client_model, body_model.usable()) {
            return ModelResolution::Conflict;
        }
    }
    // Fill by precedence. With enforcement disabled `body_model` is always
    // `Absent`, so this is exactly the legacy identity-then-client contract.
    if state.model.is_empty()
        && let Some(model) = client_model
    {
        model.clone_into(&mut state.model);
    }
    if state.model.is_empty()
        && let BodyModel::Usable(model) = &body_model
    {
        state.model.clone_from(model);
    }
    ModelResolution::Continue
}

/// Whether non-empty model sources disagree.
///
/// Trust in the identity source does not establish that the other sources
/// name the same model, so the first observed model must match every later
/// one; a disagreement means authorizing either one would admit a request
/// whose provider-bound body names a different model.
fn model_sources_conflict(identity: &str, client: Option<&str>, body: Option<&str>) -> bool {
    let sources = [(!identity.is_empty()).then_some(identity), client, body];
    let mut agreed: Option<&str> = None;
    for model in sources.into_iter().flatten() {
        match agreed {
            None => agreed = Some(model),
            Some(previous) if previous == model => {},
            Some(_) => return true,
        }
    }
    false
}

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

    let mut identity = crate::identity::resolve_tenant_identity(ctx, prefix, identity_namespace);
    strip_client_credentials(ctx);

    if identity.group.is_empty() {
        identity.group.clone_from(&identity.subscription);
    }

    MeteringState {
        cost_center: identity.cost_center,
        group: identity.group,
        is_error: false,
        model: identity.model,
        organization_id: identity.organization_id,
        request_start: std::time::Instant::now(),
        response_status: 0,
        subscription: identity.subscription,
        user_agent,
        username: identity.username,
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

/// Interpret a balance check response body against the admission policy.
fn parse_balance_result(body: &[u8], policy: AdmissionPolicy) -> FilterAction {
    decide_admission(&parse_balance_verdicts(body), policy)
}

/// Extract the per-gate verdicts from an entitlement payload.
///
/// Parsing is lenient per field: a valid `hasAccess` still decides the budget
/// gate when `modelAllowed` is missing or wrongly typed, and vice versa. A
/// body that is not a JSON object at all leaves both gates without verdicts.
fn parse_balance_verdicts(body: &[u8]) -> BalanceVerdicts {
    if body.is_empty() {
        debug!("balance check returned an empty body");
        return BalanceVerdicts::unavailable();
    }

    match serde_json::from_slice::<serde_json::Value>(body) {
        Ok(serde_json::Value::Object(fields)) => BalanceVerdicts {
            has_access: fields.get("hasAccess").and_then(serde_json::Value::as_bool),
            model_allowed: fields.get("modelAllowed").and_then(serde_json::Value::as_bool),
        },
        Ok(_) => {
            debug!("balance response was valid JSON but not an object");
            BalanceVerdicts::unavailable()
        },
        Err(e) => {
            debug!("balance response parse error: {e}");
            BalanceVerdicts::unavailable()
        },
    }
}

/// Combine both gates' verdicts into one admission decision.
///
/// Order matters: an explicit model denial wins over budget exhaustion and
/// over every availability policy, a budget denial beats every availability
/// policy, and admission on unavailable verdicts requires *every* enabled
/// gate's own fail-open setting to permit it.
fn decide_admission(verdicts: &BalanceVerdicts, policy: AdmissionPolicy) -> FilterAction {
    if policy.model_policy_check && verdicts.model_allowed == Some(false) {
        return reject_model_not_allowed();
    }

    if verdicts.has_access == Some(false) {
        return reject_budget_exhausted();
    }

    if verdicts.has_access.is_none() && !policy.billing_fail_open {
        return reject_unavailable();
    }

    if policy.model_policy_check && verdicts.model_allowed.is_none() && !policy.model_policy_fail_open {
        return reject_unavailable();
    }

    FilterAction::Continue
}

/// Reject with a 429 because the tenant has no remaining token budget.
fn reject_budget_exhausted() -> FilterAction {
    debug!("token budget exhausted");
    FilterAction::Reject(
        Rejection::status(STATUS_BUDGET_EXHAUSTED).with_body(Bytes::from_static(b"token budget exhausted")),
    )
}

/// Reject with a 403 because the requested model is not allowed for the
/// tenant. An explicit denial: returned regardless of any fail-open setting
/// and with precedence over budget exhaustion.
fn reject_model_not_allowed() -> FilterAction {
    debug!("requested model is not allowed for this user");
    FilterAction::Reject(
        Rejection::status(STATUS_MODEL_NOT_ALLOWED).with_body(Bytes::from_static(b"model not allowed for this user")),
    )
}

/// Reject with a 403 because model policy is enabled but no trusted tenant
/// identity is available to authorize. Reporting fallbacks such as
/// `default_username` never satisfy this: authorization cannot be evaluated
/// for an invented identity, so it fails closed.
fn reject_identity_required() -> FilterAction {
    debug!("model policy enabled but no tenant identity is present");
    FilterAction::Reject(
        Rejection::status(STATUS_AUTHZ_INPUT_REQUIRED)
            .with_body(Bytes::from_static(b"tenant identity required for model authorization")),
    )
}

/// Reject with a 403 because model policy is enabled but the public
/// requested model could not be resolved. `default_model` is a reporting
/// fallback only; authorizing it would not authorize the model actually
/// being requested.
fn reject_model_required() -> FilterAction {
    debug!("model policy enabled but no public model could be resolved");
    FilterAction::Reject(
        Rejection::status(STATUS_AUTHZ_INPUT_REQUIRED)
            .with_body(Bytes::from_static(b"request model required for model authorization")),
    )
}

/// Reject with a 403 because the available model sources disagree: the
/// trusted identity claim, the `model_to_provider` client-facing model, and
/// (when no client model was recorded) the request body must name the same
/// model, since authorizing one of them would not authorize the model the
/// provider actually receives.
fn reject_model_conflict() -> FilterAction {
    debug!("conflicting model inputs: identity, client, and body models disagree");
    FilterAction::Reject(
        Rejection::status(STATUS_AUTHZ_INPUT_REQUIRED)
            .with_body(Bytes::from_static(b"conflicting model inputs for model authorization")),
    )
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
/// latency to, or fail, a request the upstream already answered. `headers`
/// carries the resolved internal-auth header (empty when unauthenticated),
/// assembled on the filter before spawning so a credential gap already failed
/// the report closed by this point.
#[expect(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "sequential fire-and-forget callout setup with explicit transport inputs and the resolved auth headers; bundling them would only hide the callout's dependencies"
)]
fn spawn_usage_report(
    client: SubRequestClient,
    metering_url: &str,
    timeout: Duration,
    address_policy: AddressPolicy,
    mut headers: http::HeaderMap,
    event: &serde_json::Value,
) {
    let url = format!("{}/api/v1/events", metering_url.trim_end_matches('/'));

    let body = match serde_json::to_vec(event) {
        Ok(b) => b,
        Err(e) => {
            warn!("failed to serialize metering event: {e}");
            return;
        },
    };

    headers.insert(
        http::header::CONTENT_TYPE,
        http::HeaderValue::from_static("application/json"),
    );

    tokio::spawn(async move {
        let request = SubRequest {
            method: http::Method::POST,
            uri: http::Uri::default(),
            headers,
            body: Bytes::from(body),
        };

        // The metering service URL is operator-configured, so it resolves
        // through the shared DNS cache.
        report_delivery(
            client
                .execute_url(
                    &url,
                    request,
                    subrequest::address_policy_hook(address_policy),
                    subrequest::UrlResolutionPolicy::OperatorCached,
                    timeout,
                    MAX_CALLOUT_RESPONSE_BYTES,
                    None,
                )
                .await
                .map_err(subrequest::map_url_error),
        );
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
// Internal Auth Token (issue #1721)
// -----------------------------------------------------------------------------

/// Build the metering callout credential from the configured source.
///
/// Returns `(None, None)` when no `internal_auth_token` is configured. Otherwise
/// publishes a [`SecretHandle`] whose payload is the pre-assembled
/// `Authorization` header value, plus the file watcher (for a `file` source) the
/// caller must retain so rotation keeps publishing. An invalid initial token —
/// empty, oversized, or one that cannot form a legal header value — rejects
/// construction (fail loud at startup).
fn build_auth_token(
    source: Option<&CredentialSource>,
) -> Result<(Option<SecretHandle>, Option<WatchHandle>), FilterError> {
    let Some(source) = source else {
        return Ok((None, None));
    };
    let ctx = DiagnosticContext {
        owner: "external_metering",
        reference: None,
    };
    let transform = credential_source::bearer_header("external_metering", "internal_auth_token");
    let (handle, watch) = credential_source::build_one(ctx, source, transform)?;
    Ok((Some(handle), watch))
}

/// Resolved internal-auth state for a single metering callout.
enum AuthOutcome {
    /// No token configured; send the callout unauthenticated (legacy behavior).
    Unauthenticated,
    /// A valid token is available; attach this `Authorization` header value.
    Header(http::HeaderValue),
    /// A token is configured but currently unavailable (a post-startup rotation
    /// gap or an unusable published value); the callout must fail closed and
    /// never dial unauthenticated.
    Unavailable,
}

/// Map a published credential payload to a callout auth outcome.
///
/// `None` (a post-startup rotation gap) and a payload that cannot form a legal
/// header value both fail closed. A usable value yields a sensitive
/// `Authorization` header so the secret is never logged.
fn auth_outcome_from_payload(payload: Option<&str>) -> AuthOutcome {
    let Some(value) = payload else {
        return AuthOutcome::Unavailable;
    };
    match http::HeaderValue::from_str(value) {
        Ok(mut header) => {
            header.set_sensitive(true);
            AuthOutcome::Header(header)
        },
        Err(_) => AuthOutcome::Unavailable,
    }
}

/// Assemble the outgoing callout headers for an auth outcome.
///
/// `Some(headers)` carries the (possibly empty) header map to send; `None` means
/// fail closed — a configured token is unavailable, so the callout must not be
/// sent unauthenticated.
fn authed_headers(outcome: AuthOutcome) -> Option<http::HeaderMap> {
    let mut headers = http::HeaderMap::new();
    match outcome {
        AuthOutcome::Unauthenticated => Some(headers),
        AuthOutcome::Header(value) => {
            headers.insert(http::header::AUTHORIZATION, value);
            Some(headers)
        },
        AuthOutcome::Unavailable => None,
    }
}

/// Resolve the usage-report callout headers, or `None` to drop the report (fail
/// closed) when a configured token is unavailable, counting the drop so the
/// skipped billing event stays visible on a dashboard.
fn report_callout_headers(outcome: AuthOutcome) -> Option<http::HeaderMap> {
    let headers = authed_headers(outcome);
    if headers.is_none() {
        counter!(METRIC_REPORT_FAILURES).increment(1);
        warn!("internal metering auth token unavailable; dropping usage report");
    }
    headers
}

// -----------------------------------------------------------------------------
// URL Construction
// -----------------------------------------------------------------------------

/// A bodiless GET sub-request for the entitlement balance probe, carrying the
/// resolved internal-auth headers (empty when no token is configured).
fn balance_probe_request(headers: http::HeaderMap) -> SubRequest {
    SubRequest {
        method: http::Method::GET,
        uri: http::Uri::default(),
        headers,
        body: Bytes::new(),
    }
}

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
        data.insert("reasoning_tokens".to_owned(), tokens.reasoning.into());
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
            "organization_id": state.organization_id,
            "cost_center": state.cost_center,
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

/// Outcome of scanning a request body for its top-level `model` member.
enum BodyModel {
    /// No top-level `"model"` member was found.
    Absent,

    /// A model value the preflight and the provider read identically.
    Usable(String),

    /// A `"model"` member exists but cannot be attributed safely: duplicate
    /// or escaped key spellings, an escaped or non-string value, or a
    /// non-UTF-8 / truncated body. The provider's parser may resolve the
    /// member differently, so the value must not feed authorization.
    Unusable,
}

impl BodyModel {
    /// The usable model value, if any.
    fn usable(&self) -> Option<&str> {
        match self {
            Self::Usable(model) => Some(model),
            Self::Absent | Self::Unusable => None,
        }
    }
}

/// Extract the top-level `model` field from a JSON body fragment.
///
/// Request bodies arrive in chunks and can be megabytes long, so this scans
/// with a string- and depth-aware state machine instead of buffering and
/// deserializing the whole document. Only a `"model"` key belonging to the
/// top-level object is matched, so `"model"` occurrences nested inside
/// `messages` content can never misattribute the request. Returns
/// [`BodyModel::Absent`] when the fragment contains no top-level `"model"`
/// member, and [`BodyModel::Unusable`] when one exists but cannot be
/// attributed safely (duplicate or escaped key spellings, an escaped or
/// non-string value, or a body that is not valid UTF-8 or is cut off mid
/// value).
fn extract_model_from_bytes(bytes: &[u8]) -> BodyModel {
    let Some(text) = std::str::from_utf8(bytes).ok() else {
        return BodyModel::Unusable;
    };
    let mut scanner = TopLevelKeyScanner::new(text);
    let mut model = None;

    while let Some(key) = scanner.next_top_level_key() {
        if key_decodes_to_model(key) {
            // A second member decoding to `model` is a duplicate key: JSON
            // parsers pick their own winner, so the preflight must not guess
            // which value authorizes the request.
            if model.is_some() {
                return BodyModel::Unusable;
            }
            // Escaped values are not decoded, and the provider would parse a
            // different name than the preflight extracted: reject rather than
            // authorize under a spelling the policy never sees. The same
            // applies to a non-string value, which no provider accepts as a
            // model name.
            let Some(value) = scanner.string_value() else {
                return BodyModel::Unusable;
            };
            if value.contains('\\') {
                return BodyModel::Unusable;
            }
            model = Some(value.to_owned());
        }
    }

    match model {
        Some(model) => BodyModel::Usable(model),
        None => BodyModel::Absent,
    }
}

/// Compare a raw (still escaped) top-level object key against `"model"`.
///
/// JSON member names compare after escape decoding, so an escaped spelling
/// such as `"\u006dodel"` names the same member as `"model"`. Only a key
/// containing a backslash can decode differently from its raw text.
fn key_decodes_to_model(raw: &str) -> bool {
    if raw == "model" {
        return true;
    }
    raw.contains('\\') && serde_json::from_str::<String>(&format!("\"{raw}\"")).is_ok_and(|decoded| decoded == "model")
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
