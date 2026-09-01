// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! HTTP callout filter.
//!
//! Provides an [`HttpFilter`] that makes outbound HTTP requests during
//! request processing, extracts results from the response via `JSONPath`,
//! and feeds them into [`FilterResultSet`] for branch-chain evaluation.
//!
//! **Experimental.** Requires the `http-callout-filter` cargo feature, which
//! is off by default and activates the `experimental` marker feature. The
//! filter is a work in progress and its configuration surface may change
//! between releases.
//!
//! [`HttpFilter`]: praxis_filter::HttpFilter
//! [`FilterResultSet`]: praxis_filter::FilterResultSet

mod config;
mod extract;

#[cfg(test)]
mod tests;

use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use config::{FailureModeConfig, HttpCalloutConfig, Phase, expand_env_vars, validate_callout_url};
use extract::{BodyShaper, CompiledExtraction};
use http::HeaderMap;
use pingora_core::upstreams::peer::HttpPeer;
use praxis_core::{
    circuit::CircuitBreakerConfig as CoreCircuitBreakerConfig,
    connectivity::is_private_ip,
    subrequest::{
        DEPTH_HEADER, FrameworkHeaders, SubRequest, SubRequestClient, SubRequestConnector, SubRequestConnectorOptions,
        SubResponse,
    },
};
use praxis_filter::{
    BodyAccess, BodyMode, FilterAction, FilterError, HttpFilter, HttpFilterContext, Rejection, parse_filter_config,
};
use tracing::{debug, info, warn};

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// Filter type name.
const FILTER_NAME: &str = "http_callout";

/// Maximum allowed value for `max_body_bytes` (100 MiB).
const MAX_BODY_BYTES: usize = 104_857_600; // 100 MiB

/// Hop-by-hop and sensitive headers that must never be blindly
/// forwarded from the client onto the callout request.
const DISALLOWED_FORWARD_HEADERS: &[http::HeaderName] = &[
    http::header::HOST,
    http::header::CONTENT_LENGTH,
    http::header::TRANSFER_ENCODING,
    http::header::CONNECTION,
    http::header::UPGRADE,
    http::header::PROXY_AUTHORIZATION,
    http::header::TRAILER,
];

// -----------------------------------------------------------------------------
// HttpCalloutFilter
// -----------------------------------------------------------------------------

/// Calls an external HTTP service during request processing and feeds its response into branch-chain evaluation.
///
/// Experimental: requires the `http-callout-filter` cargo feature,
/// which is off by default and activates the `experimental` marker.
/// This filter is a work in progress and its configuration surface
/// may change between releases.
///
/// Makes an outbound HTTP request during request processing,
/// optionally forwarding the request body and downstream headers.
/// Extracts values from the callout response via `JSONPath` and
/// writes them to [`FilterResultSet`] for branch-chain evaluation.
///
/// [`FilterResultSet`]: praxis_filter::FilterResultSet
pub struct HttpCalloutFilter {
    /// Allow the target to resolve to a private/loopback/link-local
    /// address. When `false`, such a resolved peer is rejected at
    /// request time (SSRF / DNS-rebinding protection).
    allow_private_addresses: bool,

    /// Pre-compiled body shaper for reshaping the callout body.
    body_shaper: BodyShaper,

    /// Reusable HTTP client for sub-request execution.
    client: SubRequestClient,

    /// Pre-compiled `JSONPath` extraction rules.
    extractions: Vec<CompiledExtraction>,

    /// Behavior on callout failure.
    failure_mode: FailureModeConfig,

    /// Downstream headers to copy into the callout request.
    forward_headers: Vec<http::HeaderName>,

    /// Static headers to send with every callout.
    headers: Vec<(http::HeaderName, http::HeaderValue)>,

    /// Callout response headers to inject into the upstream
    /// request on success.
    inject_headers: Vec<http::HeaderName>,

    /// Maximum request body bytes to buffer.
    max_body_bytes: usize,

    /// Maximum callout depth for loop prevention.
    max_depth: u32,

    /// When the callout fires.
    phase: Phase,

    /// HTTP status code returned when rejecting on failure.
    status_on_error: u16,

    /// Parsed target (host, port, TLS, SNI, authority, request URI).
    target: CalloutTarget,

    /// Request timeout covering DNS, connect, and I/O.
    timeout: Duration,

    /// Target URL for the callout (used for logging).
    url: String,
}

impl HttpCalloutFilter {
    /// Construct the filter from a YAML config value.
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] if config parsing, SSRF validation,
    /// env-var expansion, `JSONPath` compilation, or client
    /// construction fails.
    pub fn from_config(config: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
        let cfg: HttpCalloutConfig = parse_filter_config(FILTER_NAME, config)?;

        validate_callout_url(&cfg.target.url)?;
        validate_max_body_bytes(cfg.request.max_body_bytes)?;
        validate_status_on_error(cfg.status_on_error)?;

        let body_shaper = BodyShaper::compile(&cfg.target.body)?;
        let headers = parse_static_headers(&cfg)?;
        let forward_headers = parse_header_names(&cfg.target.forward_headers, "forward_header")?;
        warn_on_disallowed_forward_headers(&forward_headers);
        let extractions = compile_extractions(&cfg)?;
        let inject_headers = parse_header_names(&cfg.response.inject_headers, "inject_header")?;

        let target = CalloutTarget::parse(&cfg.target.url)?;
        let client = build_subrequest_client(&cfg);

        Ok(Box::new(Self {
            allow_private_addresses: cfg.target.allow_private_addresses,
            body_shaper,
            client,
            extractions,
            failure_mode: cfg.on_failure,
            forward_headers,
            headers,
            inject_headers,
            max_body_bytes: cfg.request.max_body_bytes,
            max_depth: cfg.max_depth.unwrap_or(1),
            phase: cfg.request.phase,
            status_on_error: cfg.status_on_error.unwrap_or(403),
            target,
            timeout: cfg.target.timeout,
            url: cfg.target.url,
        }))
    }

    /// Build a [`SubRequest`] and [`FrameworkHeaders`] from the
    /// current filter context.
    #[expect(clippy::cast_possible_truncation, reason = "depth is clamped to u8::MAX before cast")]
    fn build_request(
        &self,
        ctx: &HttpFilterContext<'_>,
        body: Option<Vec<u8>>,
        depth: u32,
    ) -> (SubRequest, FrameworkHeaders) {
        let headers = self.build_callout_headers(ctx);

        let mut fw = FrameworkHeaders::new();
        let next_depth = (depth + 1).min(u32::from(u8::MAX));
        fw.set_depth(next_depth as u8);

        let request = SubRequest {
            method: http::Method::POST,
            uri: self.target.request_uri.clone(),
            headers,
            body: body.map_or(Bytes::new(), Bytes::from),
        };

        (request, fw)
    }

    /// Assemble the callout request headers: static configured headers,
    /// safely-forwarded client headers, and the enforced `Host`.
    fn build_callout_headers(&self, ctx: &HttpFilterContext<'_>) -> HeaderMap {
        let mut headers = HeaderMap::new();

        // Static configured headers.
        for (name, value) in &self.headers {
            headers.append(name.clone(), value.clone());
        }

        // Forward allowed client headers, skipping hop-by-hop/sensitive ones.
        for name in &self.forward_headers {
            if DISALLOWED_FORWARD_HEADERS.contains(name) {
                continue;
            }
            if let Some(value) = ctx.request.headers.get(name) {
                headers.insert(name.clone(), value.clone());
            }
        }

        // Enforce the configured target authority on the Host header.
        if let Ok(value) = self.target.authority.parse() {
            headers.insert(http::header::HOST, value);
        }

        headers
    }

    /// Process a successful callout response: extract results and
    /// inject headers.
    fn handle_success(&self, response: &SubResponse, ctx: &mut HttpFilterContext<'_>) -> FilterAction {
        if !self.extractions.is_empty() {
            if let Ok(json) = serde_json::from_slice::<serde_json::Value>(&response.body) {
                let results = ctx.filter_results.entry(self.name()).or_default();
                for extraction in &self.extractions {
                    extraction.evaluate(&json, results);
                }
                debug!(results = ?results, "extracted callout results");
            } else {
                warn!("callout response body is not valid JSON; skipping extraction");
            }
        }

        // Inject with set (overwrite) semantics: a header taken from the
        // trusted callout response replaces any client-supplied header of
        // the same name rather than being appended alongside it.
        for name in &self.inject_headers {
            if let Some(value) = response.headers.get(name) {
                ctx.request_headers_to_set.push((name.clone(), value.clone()));
            }
        }

        FilterAction::Continue
    }

    /// Build a rejection response.
    // TODO: add response headers to `core::callout::Rejection` so
    // on-denied headers can be forwarded through the callout layer.
    fn build_rejection(status: u16) -> FilterAction {
        FilterAction::Reject(Rejection::status(status))
    }

    /// Apply body shaping if configured, otherwise pass through.
    fn shape_body(&self, body: Option<Vec<u8>>) -> Option<Vec<u8>> {
        match body {
            Some(raw) if !self.body_shaper.is_empty() => {
                let result = self.body_shaper.shape(&raw);
                if result.is_none() {
                    warn!(
                        url = %self.url,
                        "body shaping failed (not valid JSON); forwarding raw body"
                    );
                }
                result.or(Some(raw))
            },
            other => other,
        }
    }

    /// Resolve DNS for the target and construct an [`HttpPeer`].
    ///
    /// When `allow_private_addresses` is `false`, the resolved peer address
    /// is validated *after* resolution against the shared classifier
    /// [`praxis_core::connectivity::is_private_ip`], so a hostname that
    /// resolves to a private/loopback/link-local address (or rebinds to one
    /// after config time) is rejected rather than connected to. Deferring to
    /// core's predicate keeps this check consistent with the rest of the
    /// proxy instead of adding another hand-rolled range list (see
    /// praxis-proxy/ai#771).
    async fn resolve_peer(&self) -> Result<HttpPeer, String> {
        let addr = tokio::net::lookup_host((self.target.host.as_str(), self.target.port))
            .await
            .map_err(|e| format!("DNS resolution failed for {}: {e}", self.target.host))?
            .next()
            .ok_or_else(|| format!("no addresses resolved for {}", self.target.host))?;

        if !self.allow_private_addresses && is_private_ip(&addr.ip()) {
            return Err(format!(
                "{} resolved to a blocked private/loopback address {} \
                 (allow_private_addresses is false)",
                self.target.host,
                addr.ip()
            ));
        }

        Ok(HttpPeer::new(
            addr.to_string(),
            self.target.tls,
            self.target.sni.clone(),
        ))
    }

    /// The action to take when the callout itself fails (DNS, connect,
    /// I/O), per the configured failure mode.
    fn failure_action(&self) -> FilterAction {
        match self.failure_mode {
            FailureModeConfig::Open => FilterAction::Continue,
            FailureModeConfig::Closed => Self::build_rejection(self.status_on_error),
        }
    }

    /// Process a completed callout response by status.
    fn handle_response(&self, response: &SubResponse, ctx: &mut HttpFilterContext<'_>) -> FilterAction {
        if !(200..300).contains(&response.status) {
            info!(url = %self.url, status = response.status, "callout rejected request");
            return Self::build_rejection(response.status);
        }

        info!(url = %self.url, status = response.status, "callout succeeded");
        self.handle_success(response, ctx)
    }

    /// Resolve the peer and perform the network round-trip.
    ///
    /// Returns the response on success, or `None` when the callout
    /// itself failed (DNS/connect/I/O) and the caller should apply
    /// [`Self::failure_action`].
    async fn perform_callout(&self, request: &SubRequest, fw: &FrameworkHeaders) -> Option<SubResponse> {
        let peer = match self.resolve_peer().await {
            Ok(p) => p,
            Err(e) => {
                warn!(url = %self.url, error = e, "callout failed");
                return None;
            },
        };

        match Box::pin(
            self.client
                .execute(&peer, request, self.max_body_bytes, self.timeout, Some(fw)),
        )
        .await
        {
            Ok(response) => Some(response),
            Err(e) => {
                warn!(url = %self.url, error = %e, "callout failed");
                None
            },
        }
    }

    /// Execute the callout and process the result.
    async fn execute_callout(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        body: Option<Vec<u8>>,
    ) -> Result<FilterAction, FilterError> {
        let depth = request_depth(ctx);
        if depth >= self.max_depth {
            info!(url = %self.url, depth, max_depth = self.max_depth, "callout depth exceeded");
            return Ok(Self::build_rejection(self.status_on_error));
        }

        let body_len = body.as_ref().map_or(0, Vec::len);
        let callout_body = self.shape_body(body);

        debug!(url = %self.url, body_bytes = body_len, "executing callout");

        let (request, fw) = self.build_request(ctx, callout_body, depth);

        let action = match Box::pin(self.perform_callout(&request, &fw)).await {
            Some(response) => self.handle_response(&response, ctx),
            None => self.failure_action(),
        };
        Ok(action)
    }
}

/// Extract the current callout depth from the framework depth header.
fn request_depth(ctx: &HttpFilterContext<'_>) -> u32 {
    ctx.request
        .headers
        .get(DEPTH_HEADER)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(0)
}

// -----------------------------------------------------------------------------
// Config Parsing Helpers
// -----------------------------------------------------------------------------

/// Reject a `max_body_bytes` value above [`MAX_BODY_BYTES`].
///
/// # Errors
///
/// Returns [`FilterError`] if `n` exceeds the allowed limit.
fn validate_max_body_bytes(n: usize) -> Result<(), FilterError> {
    if n > MAX_BODY_BYTES {
        return Err(format!("http_callout: max_body_bytes ({n}) exceeds limit ({MAX_BODY_BYTES})").into());
    }
    Ok(())
}

/// Reject a `status_on_error` value outside the valid HTTP status range.
///
/// `None` (unset) is accepted; the filter then defaults to `403`. A
/// configured value must be a legal HTTP status code (100–599) so the
/// rejection path never emits a nonsensical status like `0` or `65535`.
///
/// The `100..=599` range check is the established convention across the
/// codebase (`openai_responses_compact`, `web_search`, core builtins),
/// currently duplicated per filter. See the follow-up to promote a shared
/// `validate_status_on_error` helper into `praxis-ai-apis`.
///
/// # Errors
///
/// Returns [`FilterError`] if a configured status is outside 100–599.
fn validate_status_on_error(status: Option<u16>) -> Result<(), FilterError> {
    if let Some(code) = status
        && !(100..=599).contains(&code)
    {
        return Err(
            format!("http_callout: status_on_error ({code}) must be a valid HTTP status code (100-599)").into(),
        );
    }
    Ok(())
}

/// Parse static header entries with env-var expansion.
fn parse_static_headers(cfg: &HttpCalloutConfig) -> Result<Vec<(http::HeaderName, http::HeaderValue)>, FilterError> {
    cfg.target
        .headers
        .iter()
        .map(|h| {
            let expanded = expand_env_vars(&h.value)?;
            let name: http::HeaderName = h.name.parse().map_err(|e| -> FilterError {
                format!("http_callout: invalid header name '{}': {e}", h.name).into()
            })?;
            let value: http::HeaderValue = expanded.parse().map_err(|e| -> FilterError {
                format!("http_callout: invalid header value for '{}': {e}", h.name).into()
            })?;
            Ok((name, value))
        })
        .collect()
}

/// Parse a list of header name strings.
fn parse_header_names(names: &[String], context: &str) -> Result<Vec<http::HeaderName>, FilterError> {
    names
        .iter()
        .map(|h| {
            h.parse::<http::HeaderName>()
                .map_err(|e| -> FilterError { format!("http_callout: invalid {context} '{h}': {e}").into() })
        })
        .collect()
}

/// Warn about configured forward headers that will never be forwarded.
///
/// Hop-by-hop and sensitive headers in [`DISALLOWED_FORWARD_HEADERS`] are
/// silently skipped at request time. Surfacing them at config time tells
/// the operator their entry is a no-op instead of leaving them to wonder
/// why the header never reaches the callout target.
fn warn_on_disallowed_forward_headers(forward_headers: &[http::HeaderName]) {
    for name in forward_headers {
        if DISALLOWED_FORWARD_HEADERS.contains(name) {
            warn!(
                header = %name,
                "http_callout: forward_header '{name}' is a hop-by-hop or sensitive header \
                 and will not be forwarded to the callout target"
            );
        }
    }
}

/// Compile `JSONPath` extraction rules from config.
fn compile_extractions(cfg: &HttpCalloutConfig) -> Result<Vec<CompiledExtraction>, FilterError> {
    cfg.response
        .extract
        .iter()
        .map(|e| CompiledExtraction::compile(&e.json_path, e.result_key.clone()))
        .collect()
}

/// Parsed callout target, derived from the configured URL once and
/// reused for every callout.
#[derive(Debug)]
struct CalloutTarget {
    /// URL authority for the HTTP `Host` header (host, plus `:port`
    /// when non-default).
    authority: String,

    /// Hostname for DNS resolution.
    host: String,

    /// TCP port.
    port: u16,

    /// Path-only URI for the sub-request.
    request_uri: http::Uri,

    /// TLS SNI hostname (empty when TLS is disabled).
    sni: String,

    /// Whether TLS is enabled for the target.
    tls: bool,
}

impl CalloutTarget {
    /// Parse the target URL into the components needed for peer
    /// construction at execution time.
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] if the URL is malformed, uses a scheme
    /// other than http/https, is missing a host, or embeds userinfo.
    fn parse(url: &str) -> Result<Self, FilterError> {
        let parsed: http::Uri = url
            .parse()
            .map_err(|e| -> FilterError { format!("http_callout: invalid URL '{url}': {e}").into() })?;

        let tls = parse_scheme_tls(&parsed, url)?;
        let host = parse_host(&parsed, url)?;

        let default_port = if tls { 443 } else { 80 };
        let port = parsed
            .authority()
            .and_then(http::uri::Authority::port_u16)
            .unwrap_or(default_port);
        let sni = if tls { host.clone() } else { String::new() };
        let authority = if port == default_port {
            host.clone()
        } else {
            format!("{host}:{port}")
        };

        let path_and_query = parsed.path_and_query().map_or("/", |pq| pq.as_str());
        let request_uri: http::Uri = path_and_query
            .parse()
            .map_err(|e| -> FilterError { format!("http_callout: bad path in URL: {e}").into() })?;

        Ok(Self {
            authority,
            host,
            port,
            request_uri,
            sni,
            tls,
        })
    }
}

/// Determine whether the target scheme enables TLS (https) or not (http).
fn parse_scheme_tls(parsed: &http::Uri, url: &str) -> Result<bool, FilterError> {
    match parsed.scheme_str() {
        Some("https") => Ok(true),
        Some("http") => Ok(false),
        _ => Err(format!("http_callout: scheme must be http or https in '{url}'").into()),
    }
}

/// Extract and validate the host from a parsed target URL.
fn parse_host(parsed: &http::Uri, url: &str) -> Result<String, FilterError> {
    let authority = parsed
        .authority()
        .ok_or_else(|| -> FilterError { format!("http_callout: URL missing host: {url}").into() })?;

    // Reject userinfo (e.g. user:pass@host) to prevent credential leakage.
    if url.contains('@') {
        return Err(format!("http_callout: userinfo in URL is not allowed: {url}").into());
    }

    let host = authority
        .host()
        .trim_start_matches('[')
        .trim_end_matches(']')
        .to_owned();

    if host.is_empty() {
        return Err(format!("http_callout: empty host in URL: {url}").into());
    }

    Ok(host)
}

/// Build a [`SubRequestClient`] from parsed config.
fn build_subrequest_client(cfg: &HttpCalloutConfig) -> SubRequestClient {
    let circuit_breaker = cfg.circuit_breaker.as_ref().map(|cb| CoreCircuitBreakerConfig {
        threshold: cb.failure_threshold,
        recovery_window: cb.recovery_timeout,
        half_open_timeout: cb.recovery_timeout,
    });

    let connector = SubRequestConnector::with_options(SubRequestConnectorOptions {
        keepalive_pool_size: 16,
        max_connections: None,
        circuit_breaker,
    });

    // Cap buffered response bytes at the configured `max_body_bytes`.
    // `execute` uses `min(per_call_limit, client_ceiling)`, so leaving the
    // client at its 64 MiB default would silently clamp any configured
    // `max_body_bytes` above 64 MiB. Setting the ceiling here keeps the
    // effective response limit equal to the operator's configured value.
    SubRequestClient::with_max_response_bytes(connector, cfg.request.max_body_bytes)
}

// -----------------------------------------------------------------------------
// HttpFilter Implementation
// -----------------------------------------------------------------------------

#[async_trait]
impl HttpFilter for HttpCalloutFilter {
    fn name(&self) -> &'static str {
        // Literal required: `cargo xtask generate-filter-docs` discovers
        // filter anchors by the string literal returned from `name()`.
        // Kept in sync with `FILTER_NAME` by `name_matches_filter_name`.
        "http_callout"
    }

    fn request_body_access(&self) -> BodyAccess {
        match self.phase {
            Phase::RequestBody => BodyAccess::ReadOnly,
            Phase::RequestHeaders => BodyAccess::None,
        }
    }

    fn request_body_mode(&self) -> BodyMode {
        match self.phase {
            Phase::RequestBody => BodyMode::StreamBuffer {
                max_bytes: Some(self.max_body_bytes),
            },
            Phase::RequestHeaders => BodyMode::Stream,
        }
    }

    fn needs_request_context(&self) -> bool {
        true
    }

    async fn on_request(&self, ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        if self.phase != Phase::RequestHeaders {
            return Ok(FilterAction::Continue);
        }

        Box::pin(self.execute_callout(ctx, None)).await
    }

    async fn on_request_body(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        body: &mut Option<Bytes>,
        end_of_stream: bool,
    ) -> Result<FilterAction, FilterError> {
        if self.phase != Phase::RequestBody || !end_of_stream {
            return Ok(FilterAction::Continue);
        }

        let body_bytes = body.as_ref().map(|b| b.to_vec());
        Box::pin(self.execute_callout(ctx, body_bytes)).await
    }
}
