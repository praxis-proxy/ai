// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! `NeMo` Guardrails provider: calls `/v1/checks` and maps the response
//! to [`GuardResult`].

use std::time::{Duration, Instant};

use bytes::Bytes;
use http::{HeaderMap, HeaderValue, Method};
use praxis_ai_apis::{
    callout_target::{AddressPolicy, validate_configured_http_target, validate_resolved_addrs},
    credential_source::{self, CredentialSource, DiagnosticContext, SecretHandle, WatchHandle},
    subrequest::{SubRequest, SubRequestClient, SubResponse},
};
use praxis_core::connectivity::{PreparedTarget, UrlResolutionPolicy, UrlTargetError, prepare_url_target_with_policy};
use praxis_filter::{
    CalloutOutcome, CalloutResponse, DeferredCredential, FilterError, FilteredSubrequestExecutor, PendingCredentials,
    RequestExtensions, StagedUpstream, StagedUpstreamFallback,
};
use praxis_tls::{CaConfig, CachedClusterTls, ClusterTls};
use serde::{Deserialize, Serialize};

use super::{GuardCalloutRuntime, GuardPhase, GuardResult};

/// Default timeout for `NeMo` HTTP calls (10 seconds).
const DEFAULT_TIMEOUT_MS: u64 = 10_000;

/// Default maximum number of per-message `/v1/checks` callouts per request.
const DEFAULT_MAX_MESSAGE_CHECKS: u32 = 32;

/// Maximum response body size accepted from `NeMo` (1 MiB).
const MAX_RESPONSE_SIZE: usize = 1024 * 1024;

/// Input-only rail selection for request-phase checks.
const INPUT_RAIL_TYPES: &[&str] = &["input"];
/// Output-only rail selection for response-phase checks.
const OUTPUT_RAIL_TYPES: &[&str] = &["output"];

/// `NeMo`-specific configuration fields.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct NemoConfig {
    /// Address of the `NeMo` service. Use `https://` in production.
    ///
    /// Without `tls.ca_file`, the call checks the server certificate against
    /// the trust store loaded for the Praxis process. Set `tls.ca_file` to
    /// verify this provider's server with a private bundle instead.
    endpoint: String,

    /// Model name sent in each request. Defaults to `""` when omitted.
    #[serde(default)]
    model: String,

    /// Optional guardrail configuration selection sent to `NeMo`.
    #[serde(default)]
    guardrails: Option<NemoGuardrails>,

    /// Per-request timeout in milliseconds.
    #[serde(default = "default_timeout_ms")]
    timeout_ms: u64,

    /// Maximum number of per-message `/v1/checks` callouts per proxy request.
    #[serde(default = "default_max_message_checks")]
    max_message_checks: u32,

    /// Provider-specific trust bundle for the `NeMo` server certificate.
    #[serde(default)]
    tls: Option<NemoTls>,

    /// Path to a Kubernetes service account token.
    ///
    /// When set, each call to this `NeMo` address sends `Authorization: Bearer`
    /// with the current token. The file is read at startup and refreshed when
    /// Kubernetes rotates it. The token is added after the outbound chain
    /// runs, so a filter on that chain cannot remove it. An `http` endpoint is
    /// refused unless its host is loopback. A file that is missing, empty, or
    /// not a valid header value refuses to start. If a later reload fails, the
    /// `NeMo` call is refused and handled as a provider error.
    #[serde(default)]
    token_file: Option<String>,
}

/// PEM trust bundle for one `NeMo` peer.
///
/// The bundle replaces the process roots for this provider's upstream only.
/// Hostname verification stays enabled.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct NemoTls {
    /// Path to the PEM file of CAs that signed the `NeMo` server certificate.
    ca_file: String,
}

/// Returns the default timeout value for serde deserialization.
fn default_timeout_ms() -> u64 {
    DEFAULT_TIMEOUT_MS
}

/// Returns the default per-request message check limit for serde deserialization.
fn default_max_message_checks() -> u32 {
    DEFAULT_MAX_MESSAGE_CHECKS
}

/// Guardrail configurations selected for evaluation.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct NemoGuardrails {
    /// Configuration IDs understood by the configured `NeMo` service.
    config_ids: Vec<String>,
}

/// Phase-specific rail selection sent under `guardrails.rail_types`.
#[derive(Serialize)]
struct NemoGuardrailsRequest<'a> {
    /// Omitted when the service should select its default configuration.
    #[serde(skip_serializing_if = "Option::is_none")]
    config_ids: Option<&'a [String]>,
    /// Rail types `NeMo` should run for this check (`input` or `output`).
    rail_types: &'static [&'static str],
}

/// Outgoing request payload for `NeMo`
#[derive(Serialize)]
struct NemoRequest<'a, 'b> {
    /// Model name
    model: &'a str,
    /// List of messages to evaluate.
    messages: &'b [serde_json::Value],
    /// `NeMo` guardrails options for `/v1/checks`.
    guardrails: NemoGuardrailsRequest<'a>,
}

/// Incoming response payload for `/v1/checks`.
#[derive(Deserialize)]
struct NemoResponse {
    /// Overall verdict: `"passed"`, `"blocked"`, or `"modified"`.
    status: String,

    /// Content after rails processing. This field is required by `/v1/checks`;
    /// an empty string remains valid for a full redaction.
    content: String,

    /// Name of the blocking or modifying rail, when supplied.
    rail: Option<String>,
}

/// `NeMo` Guardrails provider.
pub(in crate::guardrails) struct NemoProvider {
    /// Shared HTTP client with admission control and circuit breaking.
    client: SubRequestClient,

    /// `NeMo` endpoint URL.
    endpoint: String,

    /// Model name included in every request. Empty string when not configured.
    model: String,

    /// Optional guardrail configuration selection.
    guardrails: Option<NemoGuardrails>,

    /// Per-request deadline covering target preparation and the outbound exchange.
    timeout: Duration,

    /// Maximum per-message `/v1/checks` callouts per proxy request.
    max_message_checks: u32,

    /// Parsed CA for this provider's HTTPS peer.
    ///
    /// `None` keeps the process trust store. The SNI applied at call time is
    /// the endpoint host.
    ca: Option<CachedClusterTls>,

    /// Service account token staged onto each call when `token_file` is set.
    service_account: Option<NemoServiceAccount>,
}

/// Pod service account token and the `NeMo` host:port allowed to receive it.
struct NemoServiceAccount {
    /// Pre-assembled `Bearer` header value. An empty snapshot refuses the call.
    header: SecretHandle,
    /// Keeps the token file watcher alive while this provider exists.
    _watch: Option<WatchHandle>,
    /// Exact `host:port` the executor resolves for `provider.endpoint`.
    authority: String,
}

impl NemoProvider {
    /// Parse and validate `NeMo`-specific config from the provider settings.
    ///
    /// Uses the provided [`SubRequestClient`] and bound outbound chain so
    /// callouts inherit the runtime's admission control, circuit breaking,
    /// and outbound security filters.
    ///
    /// # Errors
    ///
    /// Returns `FilterError` if the configuration is invalid.
    pub fn from_config(config: &serde_yaml::Value, client: SubRequestClient) -> Result<Self, FilterError> {
        let cfg: NemoConfig = serde_yaml::from_value(config.clone())
            .map_err(|e| -> FilterError { format!("ai_guardrails (nemo): {e}").into() })?;
        validate_guardrails_config(&cfg)?;
        if cfg.endpoint.is_empty() {
            return Err("ai_guardrails (nemo): 'endpoint' must not be empty".into());
        }
        validate_configured_http_target(
            "ai_guardrails (nemo)",
            &cfg.endpoint,
            // Construction validates URL shape. The bound outbound pipeline's
            // runtime policy is applied after nested pipelines are built and
            // enforced again against the pinned resolution for every callout.
            AddressPolicy::from_allow_private(true),
        )?;
        if cfg.timeout_ms == 0 {
            return Err("ai_guardrails (nemo): 'timeout_ms' must be greater than zero".into());
        }
        if cfg.max_message_checks == 0 {
            return Err("ai_guardrails (nemo): 'max_message_checks' must be greater than zero".into());
        }
        let (ca, service_account) = load_peer_material(&cfg)?;

        Ok(Self {
            client,
            endpoint: cfg.endpoint,
            model: cfg.model,
            guardrails: cfg.guardrails,
            timeout: Duration::from_millis(cfg.timeout_ms),
            max_message_checks: cfg.max_message_checks,
            ca,
            service_account,
        })
    }

    /// Return the validated timeout used for the complete provider callout.
    pub(in crate::guardrails) fn callout_timeout(&self) -> Duration {
        self.timeout
    }
}

#[expect(
    clippy::multiple_inherent_impl,
    reason = "separates construction from the async callout path"
)]
impl NemoProvider {
    /// Evaluate extracted messages against the configured `NeMo` service.
    pub(in crate::guardrails) async fn evaluate(
        &self,
        messages: Vec<serde_json::Value>,
        phase: GuardPhase,
        runtime: &GuardCalloutRuntime<'_>,
    ) -> Result<GuardResult, FilterError> {
        let indices = target_message_indices(&messages, phase);
        if indices.is_empty() {
            return Ok(GuardResult::Pass);
        }
        ensure_check_limit(indices.len(), self.max_message_checks)?;
        self.evaluate_prefixes(messages, indices, phase, runtime).await
    }

    /// Evaluate each target prefix, accumulating `modified` replacements.
    async fn evaluate_prefixes(
        &self,
        messages: Vec<serde_json::Value>,
        indices: Vec<usize>,
        phase: GuardPhase,
        runtime: &GuardCalloutRuntime<'_>,
    ) -> Result<GuardResult, FilterError> {
        let mut pending_redact = None;
        for end in indices {
            ensure_deadline_remaining(runtime.deadline)?;
            let prefix = messages
                .get(..=end)
                .ok_or_else(|| -> FilterError { "ai_guardrails (nemo): invalid message index".into() })?;
            match apply_slice_result(pending_redact, self.check_messages(prefix, phase, runtime).await?, end) {
                Ok(pending) => pending_redact = pending,
                Err(block) => return Ok(block),
            }
        }
        Ok(pending_redact.unwrap_or(GuardResult::Pass))
    }

    /// POST one message slice to `/v1/checks` through the outbound chain.
    async fn check_messages(
        &self,
        messages: &[serde_json::Value],
        phase: GuardPhase,
        runtime: &GuardCalloutRuntime<'_>,
    ) -> Result<NemoVerdict, FilterError> {
        let request = build_request(&self.model, messages, phase, self.guardrails.as_ref())?;
        let response = Box::pin(self.execute_callout(request, runtime)).await?;
        ensure_success_status(&response)?;
        let nemo_response: NemoResponse = serde_json::from_slice(&response.body).map_err(|error| -> FilterError {
            format!("ai_guardrails (nemo): failed to parse response: {error}").into()
        })?;
        map_nemo_response(nemo_response)
    }

    /// Execute one `NeMo` request through the bound outbound filter chain.
    #[expect(
        clippy::too_many_lines,
        reason = "target pinning and classified response mapping form one operation"
    )]
    #[expect(
        clippy::large_stack_frames,
        reason = "filtered subrequest executor owns the callout state machine"
    )]
    async fn execute_callout(
        &self,
        request: SubRequest,
        runtime: &GuardCalloutRuntime<'_>,
    ) -> Result<SubResponse, FilterError> {
        let mut extensions = RequestExtensions::default();
        stage_service_account_token(self.service_account.as_ref(), &mut extensions)?;

        let address_policy = AddressPolicy::from_allow_private(runtime.outbound.allow_private_upstreams());
        // The guardrails provider endpoint is operator-configured, so it
        // resolves through the shared DNS cache.
        let target = prepare_url_target_with_policy(
            &self.endpoint,
            runtime.deadline,
            |addrs| {
                validate_resolved_addrs("ai_guardrails (nemo)", addrs, address_policy)
                    .map(|_| ())
                    .map_err(|error| -> Box<dyn std::error::Error + Send + Sync> { error.to_string().into() })
            },
            UrlResolutionPolicy::OperatorCached,
        )
        .await
        .map_err(map_url_target_error)?;

        extensions.insert(stage_nemo_upstream(&target, self.ca.as_ref())?);
        if target.addresses().len() > 1 {
            extensions.insert(StagedUpstreamFallback::from_prepared_target(&target));
        }
        let prepared = target.bind(request);

        let executor = FilteredSubrequestExecutor::for_callout(
            self.client.clone(),
            runtime.downstream.clone(),
            runtime.depth,
            MAX_RESPONSE_SIZE,
            self.timeout,
        );

        match Box::pin(executor.run_classified(runtime.outbound, prepared.request(), extensions, runtime.deadline))
            .await?
        {
            CalloutOutcome::Response(CalloutResponse::Buffered(response)) => Ok(response),
            CalloutOutcome::Response(CalloutResponse::Streaming { .. }) => {
                Err("ai_guardrails (nemo): streaming NeMo responses are not supported".into())
            },
            CalloutOutcome::ResponseTooLarge { actual, limit } => Err(format!(
                "ai_guardrails (nemo): provider response exceeded size limit ({} bytes, limit {limit} bytes)",
                actual.map_or_else(|| "unknown".to_owned(), |size| size.to_string())
            )
            .into()),
            _ => Err("ai_guardrails (nemo): unsupported provider response mode".into()),
        }
    }
}

// -----------------------------------------------------------------------------
// Private Utilities
// -----------------------------------------------------------------------------

/// Load the provider CA and service account token named in the config.
///
/// # Errors
///
/// Returns [`FilterError`] when the CA file or token file cannot be read,
/// when a CA is set on a plaintext endpoint, or when a token file is set on
/// an `http` endpoint whose host is not loopback.
fn load_peer_material(cfg: &NemoConfig) -> Result<(Option<CachedClusterTls>, Option<NemoServiceAccount>), FilterError> {
    let ca = match cfg.tls.as_ref() {
        Some(tls) => {
            require_https_for_ca(&cfg.endpoint)?;
            Some(load_provider_ca(&tls.ca_file)?)
        },
        None => None,
    };
    let service_account = match cfg.token_file.as_deref() {
        Some(path) => Some(open_service_account(path, &cfg.endpoint)?),
        None => None,
    };
    Ok((ca, service_account))
}

/// Read and watch `token_file`, publishing it as a `Bearer` header value.
///
/// # Errors
///
/// Returns [`FilterError`] when the endpoint cannot carry a token, or when the
/// file is missing, empty, oversized, or does not form a valid header value.
fn open_service_account(token_file: &str, endpoint: &str) -> Result<NemoServiceAccount, FilterError> {
    require_token_transport(endpoint)?;
    let authority = praxis_ai_apis::credential_authority("ai_guardrails (nemo)", endpoint)?;
    let ctx = DiagnosticContext {
        owner: "ai_guardrails (nemo)",
        reference: None,
    };
    let source = CredentialSource::File(token_file.to_owned());
    let transform = credential_source::bearer_header("ai_guardrails (nemo)", "token_file");
    let (header, watch) = credential_source::build_one(ctx, &source, transform)?;
    Ok(NemoServiceAccount {
        header,
        _watch: watch,
        authority,
    })
}

/// Reject `tls.ca_file` on a plaintext endpoint.
///
/// # Errors
///
/// Returns [`FilterError`] when the endpoint is not `https`.
fn require_https_for_ca(endpoint: &str) -> Result<(), FilterError> {
    let url = url::Url::parse(endpoint)
        .map_err(|error| -> FilterError { format!("ai_guardrails (nemo): {error}").into() })?;
    if url.scheme() == "https" {
        return Ok(());
    }
    Err("ai_guardrails (nemo): 'tls.ca_file' requires an https endpoint".into())
}

/// Reject a service account token on plaintext except for a loopback host.
///
/// # Errors
///
/// Returns [`FilterError`] when `endpoint` is `http` and its host is not
/// loopback. `https` is always accepted.
fn require_token_transport(endpoint: &str) -> Result<(), FilterError> {
    let url = url::Url::parse(endpoint)
        .map_err(|error| -> FilterError { format!("ai_guardrails (nemo): {error}").into() })?;
    if url.scheme() == "https" || url.host().as_ref().is_some_and(host_is_loopback) {
        return Ok(());
    }
    Err("ai_guardrails (nemo): 'token_file' requires https unless the endpoint host is loopback".into())
}

/// Whether `host` is `localhost`, `127.0.0.0/8`, or `::1`.
fn host_is_loopback(host: &url::Host<&str>) -> bool {
    match host {
        url::Host::Ipv4(address) => address.is_loopback(),
        url::Host::Ipv6(address) => {
            address.is_loopback() || address.to_ipv4_mapped().is_some_and(|mapped| mapped.is_loopback())
        },
        url::Host::Domain(domain) => domain.eq_ignore_ascii_case("localhost"),
    }
}

/// Load `tls.ca_file` as this peer's only trust anchors.
///
/// Verification stays enabled. The caller sets SNI from the endpoint host.
///
/// # Errors
///
/// Returns [`FilterError`] when the path is empty, traverses a parent
/// directory, or the PEM cannot be read.
fn load_provider_ca(ca_file: &str) -> Result<CachedClusterTls, FilterError> {
    if ca_file.is_empty() {
        return Err("ai_guardrails (nemo): 'tls.ca_file' must not be empty".into());
    }
    let tls = ClusterTls {
        ca: Some(CaConfig {
            ca_path: ca_file.to_owned(),
            crl_paths: Vec::new(),
        }),
        client_cert: None,
        sni: None,
        verify: true,
    };
    tls.validate()
        .map_err(|error| -> FilterError { format!("ai_guardrails (nemo): invalid tls.ca_file: {error}").into() })?;
    CachedClusterTls::try_from_config(&tls)
        .map_err(|error| -> FilterError { format!("ai_guardrails (nemo): failed to load tls.ca_file: {error}").into() })
}

/// Stage the resolved `NeMo` peer, attaching the provider CA when configured.
///
/// The CA replaces the process roots for this upstream. The SNI and hostname
/// check stay those of the prepared target.
///
/// # Errors
///
/// Returns [`FilterError`] when the target has no TLS material to verify.
fn stage_nemo_upstream(target: &PreparedTarget, ca: Option<&CachedClusterTls>) -> Result<StagedUpstream, FilterError> {
    let mut staged = StagedUpstream::from_prepared_target(target)?;
    let Some(ca) = ca else {
        return Ok(staged);
    };
    let sni = staged
        .0
        .tls
        .as_ref()
        .and_then(|existing| existing.sni().map(str::to_owned))
        .ok_or_else(|| -> FilterError { "ai_guardrails (nemo): 'tls.ca_file' requires an https endpoint".into() })?;
    let mut tls = ca.clone();
    tls.set_sni(sni);
    staged.0.tls = Some(tls);
    Ok(staged)
}

/// Reject an explicitly empty configuration selection.
fn validate_guardrails_config(config: &NemoConfig) -> Result<(), FilterError> {
    if config
        .guardrails
        .as_ref()
        .is_some_and(|guardrails| guardrails.config_ids.is_empty())
    {
        return Err(
            "ai_guardrails (nemo): 'guardrails.config_ids' must not be empty; omit 'guardrails' to use the service default"
                .into(),
        );
    }
    Ok(())
}

/// Reject conversations that would exceed the configured per-request check cap.
fn ensure_check_limit(count: usize, max_message_checks: u32) -> Result<(), FilterError> {
    if count > max_message_checks as usize {
        return Err(format!(
            "ai_guardrails (nemo): conversation has {count} target messages, exceeding max_message_checks ({max_message_checks})"
        )
        .into());
    }
    Ok(())
}

/// Fail closed if the overall evaluation deadline has already elapsed.
fn ensure_deadline_remaining(deadline: Instant) -> Result<(), FilterError> {
    deadline
        .checked_duration_since(Instant::now())
        .map(|_| ())
        .ok_or_else(|| "ai_guardrails (nemo): overall evaluation deadline exceeded".into())
}

/// Combine a per-slice verdict into the running evaluation state.
///
/// `blocked` fails fast. `modified` replacements are accumulated so every
/// redacted turn is rewritten. Later slices are still checked so a subsequent
/// `blocked` verdict is not missed. The replacement index is the conversation
/// index of this slice; `/v1/checks` does not report one.
fn apply_slice_result(
    pending_redact: Option<GuardResult>,
    result: NemoVerdict,
    message_index: usize,
) -> Result<Option<GuardResult>, GuardResult> {
    match result {
        NemoVerdict::Pass => Ok(pending_redact),
        NemoVerdict::Block { reason } => Err(GuardResult::Block { reason }),
        NemoVerdict::Modified { content, reason } => Ok(Some(merge_redact(
            pending_redact,
            GuardResult::redact_message(message_index, content, reason),
        ))),
    }
}

/// Append one `modified` verdict onto any pending redaction.
///
/// A later `modified` slice replaces the reason and keeps every earlier replacement.
fn merge_redact(pending: Option<GuardResult>, extra: GuardResult) -> GuardResult {
    let Some(GuardResult::Redact {
        replacements: mut existing,
        ..
    }) = pending
    else {
        return extra;
    };
    let GuardResult::Redact {
        replacements: extra_replacements,
        reason,
    } = extra
    else {
        return extra;
    };
    existing.extend(extra_replacements);
    GuardResult::Redact {
        replacements: existing,
        reason,
    }
}

/// Indices of messages checked for the active guard phase.
///
/// `/v1/checks` evaluates only the last message of a slice. Request phase
/// issues one call per user turn, each carrying the conversation prefix up to
/// that turn. Response `choices` are alternative completions, not turns, so
/// only `choices[0].message` is checked and redacted.
fn target_message_indices(messages: &[serde_json::Value], phase: GuardPhase) -> Vec<usize> {
    match phase {
        GuardPhase::Request => messages
            .iter()
            .enumerate()
            .filter(|(_, message)| message_role(message) == Some("user"))
            .map(|(index, _)| index)
            .collect(),
        GuardPhase::Response => match messages.first() {
            Some(message) if message_role(message) == Some("assistant") => vec![0],
            _ => Vec::new(),
        },
    }
}

/// Only a string `role` can select a target turn. A missing or non-string role is skipped.
fn message_role(message: &serde_json::Value) -> Option<&str> {
    message.get("role").and_then(serde_json::Value::as_str)
}

/// `NeMo` rail types for the active guard phase.
fn rail_types_for_phase(phase: GuardPhase) -> &'static [&'static str] {
    match phase {
        GuardPhase::Request => INPUT_RAIL_TYPES,
        GuardPhase::Response => OUTPUT_RAIL_TYPES,
    }
}

/// Build the outbound `NeMo` JSON callout.
fn build_request<'a>(
    model: &'a str,
    messages: &[serde_json::Value],
    phase: GuardPhase,
    guardrails: Option<&'a NemoGuardrails>,
) -> Result<SubRequest, FilterError> {
    let payload = NemoRequest {
        model,
        messages,
        guardrails: NemoGuardrailsRequest {
            config_ids: guardrails.map(|settings| settings.config_ids.as_slice()),
            rail_types: rail_types_for_phase(phase),
        },
    };
    let body =
        Bytes::from(serde_json::to_vec(&payload).map_err(|e| -> FilterError {
            format!("ai_guardrails (nemo): failed to serialize request: {e}").into()
        })?);

    let mut headers = HeaderMap::new();
    headers.insert(http::header::CONTENT_TYPE, HeaderValue::from_static("application/json"));
    headers.insert(http::header::ACCEPT, HeaderValue::from_static("application/json"));

    Ok(SubRequest {
        method: Method::POST,
        uri: http::Uri::default(),
        headers,
        body,
    })
}

/// Stage the service account token for injection after the outbound chain.
///
/// The executor writes `Authorization` only once the dial is pinned to
/// `authority`. The header is not on the request that chain sees, so a filter
/// there cannot remove or replace it. A configured token that cannot be read
/// refuses the call before anything is sent.
///
/// # Errors
///
/// Returns [`FilterError`] when a token file is configured and the latest
/// reload produced no usable token.
fn stage_service_account_token(
    account: Option<&NemoServiceAccount>,
    extensions: &mut RequestExtensions,
) -> Result<(), FilterError> {
    let Some(account) = account else {
        return Ok(());
    };
    let snapshot = account.header.current();
    let Some(value) = snapshot.payload.as_deref() else {
        tracing::debug!("ai_guardrails (nemo): service account token unavailable");
        return Err("ai_guardrails (nemo): service account token unavailable".into());
    };
    let credential = DeferredCredential::new(&account.authority, http::header::AUTHORIZATION, value.as_str())?;
    let mut pending = PendingCredentials::new();
    pending.push(credential);
    extensions.insert(pending);
    Ok(())
}

/// Map URL target preparation failures to filter errors.
fn map_url_target_error(error: UrlTargetError) -> FilterError {
    match error {
        UrlTargetError::InvalidTarget(invalid) => {
            format!("ai_guardrails (nemo): invalid endpoint URL: {invalid}").into()
        },
        UrlTargetError::Resolve(resolve) => {
            format!("ai_guardrails (nemo): failed to resolve endpoint: {resolve}").into()
        },
        UrlTargetError::PolicyRejected(source) => {
            format!("ai_guardrails (nemo): endpoint address policy rejected target: {source}").into()
        },
        UrlTargetError::DeadlineExceeded => "ai_guardrails (nemo): endpoint preparation deadline exceeded".into(),
        _ => "ai_guardrails (nemo): endpoint preparation failed".into(),
    }
}

/// Reject non-2xx HTTP responses from the provider.
fn ensure_success_status(response: &SubResponse) -> Result<(), FilterError> {
    if !(200..300).contains(&response.status) {
        return Err(format!(
            "ai_guardrails (nemo): provider returned HTTP status code {}",
            response.status
        )
        .into());
    }
    Ok(())
}

/// One `/v1/checks` verdict before it is attached to a conversation index.
///
/// `modified` carries the masked text and rail name only. The message index
/// belongs to the slice the caller selected.
#[derive(Debug, Clone, PartialEq, Eq)]
enum NemoVerdict {
    /// All rails passed.
    Pass,
    /// At least one rail blocked the content.
    Block {
        /// Human-readable block reason from the provider.
        reason: String,
    },
    /// Content was transformed, for example by PII masking.
    Modified {
        /// Masked text returned in the `content` field.
        content: String,
        /// Human-readable redaction reason from the provider.
        reason: String,
    },
}

/// Map a deserialized [`NemoResponse`] to a [`NemoVerdict`].
///
/// The `/v1/checks` endpoint returns three statuses:
/// - `"passed"` - all rails passed
/// - `"blocked"` - at least one rail blocked the content
/// - `"modified"` - content was transformed (for example, PII masking)
fn map_nemo_response(nemo: NemoResponse) -> Result<NemoVerdict, FilterError> {
    let NemoResponse { status, content, rail } = nemo;
    match status.as_str() {
        "passed" => Ok(NemoVerdict::Pass),
        "blocked" => Ok(NemoVerdict::Block {
            reason: rail.unwrap_or_default(),
        }),
        "modified" => Ok(NemoVerdict::Modified {
            content,
            reason: rail.unwrap_or_else(|| "modified".to_owned()),
        }),
        other => Err(format!("ai_guardrails (nemo): unknown status '{other}'").into()),
    }
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(
    clippy::unwrap_used,
    clippy::str_to_string,
    clippy::indexing_slicing,
    reason = "tests"
)]
mod tests {
    use zeroize::Zeroizing;

    use super::{super::MessageRedaction, *};

    fn fixed_account(payload: Option<&str>) -> NemoServiceAccount {
        NemoServiceAccount {
            header: SecretHandle::new(credential_source::SecretSnapshot {
                payload: payload.map(|value| Zeroizing::new(value.to_owned())),
            }),
            _watch: None,
            authority: "127.0.0.1:3001".to_owned(),
        }
    }

    fn write_token(contents: &str) -> (tempfile::TempDir, String) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("token");
        std::fs::write(&path, contents).unwrap();
        (dir, path.to_str().unwrap().to_owned())
    }

    #[test]
    fn available_token_is_staged_as_a_deferred_credential() {
        let account = fixed_account(Some("Bearer sa-jwt"));
        let mut extensions = RequestExtensions::default();
        stage_service_account_token(Some(&account), &mut extensions).unwrap();
        let pending = extensions.get::<PendingCredentials>().unwrap();
        assert!(!pending.is_empty(), "credential staged");
    }

    #[test]
    fn token_file_is_published_as_a_bearer_header_value() {
        let (_dir, path) = write_token("sa-jwt\n");
        let account = open_service_account(&path, "http://127.0.0.1:3001/v1/checks").unwrap();
        assert_eq!(
            account.header.current().payload.as_deref().map(String::as_str),
            Some("Bearer sa-jwt")
        );
        assert_eq!(account.authority, "127.0.0.1:3001");
    }

    #[test]
    fn token_that_cannot_be_a_header_value_is_rejected_at_startup() {
        let (_dir, path) = write_token("sa\u{7f}jwt\n");
        let err = open_service_account(&path, "http://127.0.0.1:3001/v1/checks")
            .err()
            .unwrap();
        assert!(err.to_string().contains("not a valid header value"), "{err}");
    }

    #[test]
    fn token_file_http_requires_a_loopback_host() {
        assert!(require_token_transport("https://nemo.default.svc:8443/v1/checks").is_ok());
        assert!(require_token_transport("http://127.0.0.1:3001/v1/checks").is_ok());
        assert!(require_token_transport("http://localhost:3001/v1/checks").is_ok());
        assert!(require_token_transport("http://[::1]:3001/v1/checks").is_ok());
        assert!(require_token_transport("http://[::ffff:127.0.0.1]:3001/v1/checks").is_ok());
        let err = require_token_transport("http://nemo:8000/v1/checks").unwrap_err();
        assert!(err.to_string().contains("loopback"), "{err}");
    }

    #[test]
    fn omitted_token_adds_no_authorization() {
        let mut extensions = RequestExtensions::default();
        stage_service_account_token(None, &mut extensions).unwrap();
        assert!(extensions.get::<PendingCredentials>().is_none());
    }

    #[test]
    fn unavailable_token_refuses_the_call() {
        let account = fixed_account(None);
        let mut extensions = RequestExtensions::default();
        let err = stage_service_account_token(Some(&account), &mut extensions).unwrap_err();
        assert!(err.to_string().contains("unavailable"), "{err}");
        assert!(extensions.get::<PendingCredentials>().is_none());
    }

    #[test]
    fn request_serializes_configured_guardrails() {
        let config: NemoConfig = serde_yaml::from_str(
            r#"
endpoint: "http://localhost:8000/v1/checks"
model: check-model
guardrails:
  config_ids: [your-config, another-config]
"#,
        )
        .unwrap();
        let messages = vec![serde_json::json!({"role": "user", "content": "Hello"})];
        let request = build_request(
            &config.model,
            &messages,
            GuardPhase::Request,
            config.guardrails.as_ref(),
        )
        .unwrap();

        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(&request.body).unwrap(),
            serde_json::json!({
                "model": "check-model",
                "messages": [{"role": "user", "content": "Hello"}],
                "guardrails": {"rail_types": ["input"], "config_ids": ["your-config", "another-config"]}
            })
        );
    }

    #[test]
    fn request_omits_unconfigured_config_ids() {
        let config: NemoConfig = serde_yaml::from_str("endpoint: http://localhost:8000").unwrap();
        let request = build_request(&config.model, &[], GuardPhase::Response, config.guardrails.as_ref()).unwrap();

        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(&request.body).unwrap(),
            serde_json::json!({
                "model": "", "messages": [], "guardrails": {"rail_types": ["output"]}
            })
        );
    }

    #[test]
    fn target_message_indices_request_collects_user_turns() {
        let messages = vec![
            serde_json::json!({"role": "system", "content": "sys"}),
            serde_json::json!({"role": "user", "content": "first"}),
            serde_json::json!({"role": "assistant", "content": "reply"}),
            serde_json::json!({"role": "user", "content": "second"}),
        ];

        assert_eq!(target_message_indices(&messages, GuardPhase::Request), vec![1, 3]);
    }

    #[test]
    fn target_message_indices_response_checks_only_first_choice() {
        let messages = vec![
            serde_json::json!({"role": "assistant", "content": "a"}),
            serde_json::json!({"role": "assistant", "content": "b"}),
        ];

        assert_eq!(
            target_message_indices(&messages, GuardPhase::Response),
            vec![0],
            "response phase should check only the first choice"
        );
    }

    #[test]
    fn target_message_indices_response_skips_when_first_choice_is_not_assistant() {
        let messages = vec![
            serde_json::json!({"role": "user", "content": "echo"}),
            serde_json::json!({"role": "assistant", "content": "later"}),
        ];

        assert!(
            target_message_indices(&messages, GuardPhase::Response).is_empty(),
            "response phase should skip when the first choice is not an assistant message"
        );
    }

    #[test]
    fn target_message_indices_empty_when_no_target_role() {
        let messages = vec![serde_json::json!({"role": "system", "content": "sys"})];
        assert!(target_message_indices(&messages, GuardPhase::Request).is_empty());
    }

    #[test]
    fn rail_types_are_phase_specific() {
        assert_eq!(rail_types_for_phase(GuardPhase::Request), ["input"]);
        assert_eq!(rail_types_for_phase(GuardPhase::Response), ["output"]);
    }

    #[test]
    fn build_request_includes_phase_specific_rail_types() {
        let messages = vec![serde_json::json!({"role": "user", "content": "hello"})];
        let request = build_request("test", &messages, GuardPhase::Request, None).unwrap();
        let payload: serde_json::Value = serde_json::from_slice(&request.body).unwrap();

        assert_eq!(payload["guardrails"]["rail_types"], serde_json::json!(["input"]));
        assert_eq!(payload["messages"], serde_json::Value::Array(messages));
    }

    #[test]
    fn apply_slice_result_modified_then_blocked_fails_fast_on_blocked() {
        let modified = NemoVerdict::Modified {
            content: "masked".into(),
            reason: "pii".into(),
        };
        let blocked = NemoVerdict::Block {
            reason: "toxicity".into(),
        };

        let pending = apply_slice_result(None, modified, 1).unwrap();
        assert_eq!(
            pending,
            Some(GuardResult::redact_message(1, "masked".into(), "pii".into()))
        );
        assert_eq!(
            apply_slice_result(pending, blocked, 3).unwrap_err(),
            GuardResult::Block {
                reason: "toxicity".into(),
            }
        );
    }

    #[test]
    fn apply_slice_result_accumulates_modified_replacements() {
        let first = NemoVerdict::Modified {
            content: "first".into(),
            reason: "pii".into(),
        };
        let second = NemoVerdict::Modified {
            content: "second".into(),
            reason: "pii".into(),
        };

        let pending = apply_slice_result(None, first, 1).unwrap();
        let pending = apply_slice_result(pending, NemoVerdict::Pass, 2).unwrap();
        assert_eq!(
            apply_slice_result(pending, second, 3).unwrap(),
            Some(GuardResult::Redact {
                replacements: vec![
                    MessageRedaction {
                        index: 1,
                        modified_text: "first".into(),
                    },
                    MessageRedaction {
                        index: 3,
                        modified_text: "second".into(),
                    },
                ],
                reason: "pii".into(),
            })
        );
    }

    #[test]
    fn map_nemo_response_passed_returns_pass() {
        let resp = NemoResponse {
            status: "passed".to_string(),
            content: "hello".to_string(),
            rail: None,
        };
        let result = map_nemo_response(resp).unwrap();
        assert!(matches!(result, NemoVerdict::Pass), "passed status should map to Pass");
    }

    #[test]
    fn map_nemo_response_blocked_returns_block_with_rail() {
        let resp = NemoResponse {
            status: "blocked".to_string(),
            content: "blocked text".to_string(),
            rail: Some("toxicity".to_string()),
        };
        let result = map_nemo_response(resp).unwrap();
        assert!(
            matches!(result, NemoVerdict::Block { reason } if reason == "toxicity"),
            "blocked response should produce a block verdict with the rail name as reason"
        );
    }

    #[test]
    fn map_nemo_response_blocked_without_rail_returns_empty_reason() {
        let resp = NemoResponse {
            status: "blocked".to_string(),
            content: "blocked text".to_string(),
            rail: None,
        };
        let result = map_nemo_response(resp).unwrap();
        assert!(
            matches!(result, NemoVerdict::Block { reason } if reason.is_empty()),
            "blocked status without a rail should use an empty reason"
        );
    }

    #[test]
    fn map_nemo_response_modified_returns_content_and_reason() {
        let resp = NemoResponse {
            status: "modified".to_string(),
            content: "masked text".to_string(),
            rail: Some("pii".to_string()),
        };
        assert_eq!(
            map_nemo_response(resp).unwrap(),
            NemoVerdict::Modified {
                content: "masked text".to_string(),
                reason: "pii".to_string(),
            },
            "modified status should return the masked content and rail name"
        );
    }

    #[test]
    fn deserialize_response_requires_content_but_allows_empty_content() {
        let missing = serde_json::from_value::<NemoResponse>(serde_json::json!({"status": "passed"}));
        assert!(missing.is_err(), "missing /v1/checks content must fail closed");

        let empty = serde_json::from_value::<NemoResponse>(serde_json::json!({
            "status": "modified",
            "content": ""
        }))
        .unwrap();
        assert!(empty.content.is_empty());
    }

    #[test]
    fn map_nemo_response_unknown_status_returns_error() {
        let resp = NemoResponse {
            status: "garbage".to_string(),
            content: String::new(),
            rail: None,
        };
        let err_msg = format!("{}", map_nemo_response(resp).unwrap_err());
        assert!(
            err_msg.contains("unknown status 'garbage'"),
            "unknown status should produce a descriptive error: {err_msg}"
        );
    }

    #[test]
    fn ca_file_is_optional() {
        let config: NemoConfig = serde_yaml::from_str("endpoint: http://localhost:8000").unwrap();
        assert!(config.tls.is_none());
    }

    #[test]
    fn ca_file_rejects_unknown_tls_fields() {
        let error = serde_yaml::from_str::<NemoConfig>(
            "endpoint: https://nemo.example/v1/checks\ntls:\n  ca_file: /tmp/ca.crt\n  extra: true\n",
        )
        .unwrap_err();
        let message = error.to_string();
        assert!(
            message.contains("extra") || message.contains("unknown"),
            "unknown tls field should be rejected: {message}"
        );
    }

    #[test]
    fn ca_file_requires_https() {
        let error = require_https_for_ca("http://nemo.example/v1/checks").unwrap_err();
        assert!(format!("{error}").contains("https"));
        require_https_for_ca("https://nemo.example/v1/checks").unwrap();
    }

    #[test]
    fn ca_file_loads_pem_and_keeps_verification() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("ca.crt");
        let key = rcgen::KeyPair::generate().unwrap();
        let mut params = rcgen::CertificateParams::new(Vec::<String>::new()).unwrap();
        params.is_ca = rcgen::IsCa::Ca(rcgen::BasicConstraints::Unconstrained);
        params.distinguished_name.push(rcgen::DnType::CommonName, "test-ca");
        let certificate = params.self_signed(&key).unwrap();
        std::fs::write(&path, certificate.pem()).unwrap();
        let cached = load_provider_ca(path.to_str().unwrap()).unwrap();
        assert!(cached.verify(), "hostname verification stays enabled");
        assert!(cached.ca().is_some(), "the provider bundle is this peer's trust store");
        assert!(cached.sni().is_none(), "SNI is applied from the endpoint at call time");
    }

    #[test]
    fn ca_file_rejects_missing_file() {
        let error = load_provider_ca("/tmp/does-not-exist-nemo-ca.pem").unwrap_err();
        assert!(format!("{error}").contains("tls.ca_file"));
    }

    #[test]
    fn ca_file_rejects_parent_directory() {
        let error = load_provider_ca("/tmp/../etc/passwd").unwrap_err();
        assert!(format!("{error}").contains("tls.ca_file"));
    }

    #[test]
    fn ca_file_rejects_empty_path() {
        let error = load_provider_ca("").unwrap_err();
        assert!(format!("{error}").contains("must not be empty"));
    }
}
