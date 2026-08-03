// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! HTTP callout filter.
//!
//! Provides an [`HttpFilter`] that makes outbound HTTP requests during
//! request processing, extracts results from the response via `JSONPath`,
//! and feeds them into [`FilterResultSet`] for branch-chain evaluation.
//!
//! [`HttpFilter`]: praxis_filter::HttpFilter
//! [`FilterResultSet`]: praxis_filter::FilterResultSet

mod config;
mod extract;

#[cfg(test)]
mod tests;

use std::borrow::Cow;
use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use config::{FailureModeConfig, HttpCalloutConfig, Phase, expand_env_vars, validate_callout_url};
use extract::{BodyShaper, CompiledExtraction};
use http::HeaderMap;
use pingora_core::upstreams::peer::HttpPeer;
use praxis_core::circuit::CircuitBreakerConfig as CoreCircuitBreakerConfig;
use praxis_core::subrequest::{
    FrameworkHeaders, SubRequest, SubRequestClient, SubRequestConnector, SubRequestConnectorOptions, SubResponse,
    DEPTH_HEADER,
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

// -----------------------------------------------------------------------------
// HttpCalloutFilter
// -----------------------------------------------------------------------------

/// HTTP callout filter.
///
/// Makes an outbound HTTP request during request processing,
/// optionally forwarding the request body and downstream headers.
/// Extracts values from the callout response via `JSONPath` and
/// writes them to [`FilterResultSet`] for branch-chain evaluation.
///
/// [`FilterResultSet`]: praxis_filter::FilterResultSet
pub struct HttpCalloutFilter {
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

    /// Path-only URI for the sub-request (parsed from the target
    /// URL at config time).
    request_uri: http::Uri,

    /// HTTP status code returned when rejecting on failure.
    status_on_error: u16,

    /// Original URL authority for the HTTP `Host` header.
    target_authority: String,

    /// Hostname for DNS resolution.
    target_host: String,

    /// TCP port.
    target_port: u16,

    /// TLS SNI hostname (empty when TLS is disabled).
    target_sni: String,

    /// Whether TLS is enabled for the target.
    target_tls: bool,

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

        if cfg.request.max_body_bytes > MAX_BODY_BYTES {
            return Err(format!(
                "http_callout: max_body_bytes ({}) exceeds limit ({})",
                cfg.request.max_body_bytes, MAX_BODY_BYTES,
            )
            .into());
        }

        let body_shaper = BodyShaper::compile(&cfg.target.body)?;
        let headers = parse_static_headers(&cfg)?;
        let forward_headers = parse_header_names(&cfg.target.forward_headers, "forward_header")?;
        let extractions = compile_extractions(&cfg)?;
        let inject_headers = parse_header_names(&cfg.response.inject_headers, "inject_header")?;

        let (target_host, target_port, target_tls, target_sni, target_authority, request_uri) =
            parse_callout_target(&cfg.target.url)?;
        let client = build_subrequest_client(&cfg);

        Ok(Box::new(Self {
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
            request_uri,
            status_on_error: cfg.status_on_error.unwrap_or(403),
            target_authority,
            target_host,
            target_port,
            target_sni,
            target_tls,
            timeout: cfg.target.timeout,
            url: cfg.target.url,
        }))
    }

    /// Build a [`SubRequest`] and [`FrameworkHeaders`] from the
    /// current filter context.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "depth is clamped to u8::MAX before cast"
    )]
    fn build_request(
        &self,
        ctx: &HttpFilterContext<'_>,
        body: Option<Vec<u8>>,
        depth: u32,
    ) -> (SubRequest, FrameworkHeaders) {
        // Standard hop-by-hop and sensitive headers that should NEVER be blindly forwarded from clients
        const DISALLOWED_FORWARD_HEADERS: &[http::HeaderName] = &[
            http::header::HOST,
            http::header::CONTENT_LENGTH,
            http::header::TRANSFER_ENCODING,
            http::header::CONNECTION,
            http::header::UPGRADE,
            http::header::PROXY_AUTHORIZATION,
            http::header::TRAILER,
        ];

        let mut headers = HeaderMap::new();

        // 1. Populate static configured headers (from self.headers)
        for (name, value) in &self.headers {
            headers.append(name.clone(), value.clone());
        }

        // 2. Forward allowed client headers safely
        for name in &self.forward_headers {
            // Strip sensitive/hop-by-hop headers from forward_headers whitelist
            if DISALLOWED_FORWARD_HEADERS.contains(name) {
                continue;
            }

            if let Some(value) = ctx.request.headers.get(name) {
                headers.insert(name.clone(), value.clone());
            }
        }

        // 3. Unconditionally enforce configured target_authority on the Host header
        if let Ok(value) = self.target_authority.parse() {
            headers.insert(http::header::HOST, value);
        }

        // 4. Build framework depth context
        let mut fw = FrameworkHeaders::new();
        let next_depth = (depth + 1).min(u32::from(u8::MAX));
        fw.set_depth(next_depth as u8);

        let request = SubRequest {
            method: http::Method::POST,
            uri: self.request_uri.clone(),
            headers,
            body: body.map_or(Bytes::new(), Bytes::from),
        };

        (request, fw)
    }

    /// Process a successful callout response: extract results and
    /// inject headers.
    fn handle_success(
        &self,
        response: &SubResponse,
        ctx: &mut HttpFilterContext<'_>,
    ) -> Result<FilterAction, FilterError> {
        if !self.extractions.is_empty() {
            if let Ok(json) = serde_json::from_slice::<serde_json::Value>(&response.body) {
                let results = ctx.filter_results.entry(self.name()).or_default();
                for extraction in &self.extractions {
                    extraction.evaluate(&json, results)?;
                }
                debug!(results = ?results, "extracted callout results");
            } else {
                warn!("callout response body is not valid JSON; skipping extraction");
            }
        }

        for name in &self.inject_headers {
            if let Some(value) = response.headers.get(name)
                && let Ok(value_str) = value.to_str()
            {
                ctx.extra_request_headers
                    .push((Cow::Owned(name.to_string()), value_str.to_owned()));
            }
        }

        Ok(FilterAction::Continue)
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
    async fn resolve_peer(&self) -> Result<HttpPeer, String> {
        let addr = tokio::net::lookup_host((self.target_host.as_str(), self.target_port))
            .await
            .map_err(|e| format!("DNS resolution failed for {}: {e}", self.target_host))?
            .next()
            .ok_or_else(|| format!("no addresses resolved for {}", self.target_host))?;

        Ok(HttpPeer::new(addr.to_string(), self.target_tls, self.target_sni.clone()))
    }

    /// Execute the callout and process the result.
    async fn execute_callout(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        body: Option<Vec<u8>>,
    ) -> Result<FilterAction, FilterError> {
        let depth = ctx
            .request
            .headers
            .get(DEPTH_HEADER)
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse::<u32>().ok())
            .unwrap_or(0);

        if depth >= self.max_depth {
            info!(url = %self.url, depth, max_depth = self.max_depth, "callout depth exceeded");
            return Ok(Self::build_rejection(self.status_on_error));
        }

        let body_len = body.as_ref().map_or(0, Vec::len);
        let callout_body = self.shape_body(body);

        debug!(url = %self.url, body_bytes = body_len, "executing callout");

        let (request, fw) = self.build_request(ctx, callout_body, depth);

        let peer = match self.resolve_peer().await {
            Ok(p) => p,
            Err(e) => {
                warn!(url = %self.url, error = e, "callout failed");
                return match self.failure_mode {
                    FailureModeConfig::Open => Ok(FilterAction::Continue),
                    FailureModeConfig::Closed => Ok(Self::build_rejection(self.status_on_error)),
                };
            },
        };

        let response = match self
            .client
            .execute(&peer, &request, self.max_body_bytes, self.timeout, Some(&fw))
            .await
        {
            Ok(resp) => resp,
            Err(e) => {
                warn!(url = %self.url, error = %e, "callout failed");
                return match self.failure_mode {
                    FailureModeConfig::Open => Ok(FilterAction::Continue),
                    FailureModeConfig::Closed => Ok(Self::build_rejection(self.status_on_error)),
                };
            },
        };

        if !(200..300).contains(&response.status) {
            info!(url = %self.url, status = response.status, "callout rejected request");
            return Ok(Self::build_rejection(response.status));
        }

        info!(url = %self.url, status = response.status, "callout succeeded");
        self.handle_success(&response, ctx)
    }
}

// -----------------------------------------------------------------------------
// Config Parsing Helpers
// -----------------------------------------------------------------------------

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

/// Compile `JSONPath` extraction rules from config.
fn compile_extractions(cfg: &HttpCalloutConfig) -> Result<Vec<CompiledExtraction>, FilterError> {
    cfg.response
        .extract
        .iter()
        .map(|e| CompiledExtraction::compile(&e.json_path, e.result_key.clone()))
        .collect()
}

/// Parse the target URL into components needed for peer
/// construction at execution time.
fn parse_callout_target(
    url: &str,
) -> Result<(String, u16, bool, String, String, http::Uri), FilterError> {
    let parsed: http::Uri = url
        .parse()
        .map_err(|e| -> FilterError { format!("http_callout: invalid URL '{url}': {e}").into() })?;

    let tls = match parsed.scheme_str() {
        Some("https") => true,
        Some("http") => false,
        _ => return Err(format!("http_callout: scheme must be http or https in '{url}'").into()),
    };

    let authority = parsed
        .authority()
        .ok_or_else(|| -> FilterError { format!("http_callout: URL missing host: {url}").into() })?;

    //reject userinfo (e.g., user:pass@host) to prevent credential leakage
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

    let default_port = if tls { 443 } else { 80 };
    let port = authority.port_u16().unwrap_or(default_port);
    let sni = if tls { host.clone() } else { String::new() };
    let authority_str = if port == default_port {
        host.clone()
    } else {
        format!("{host}:{port}")
    };

    let path_and_query = parsed.path_and_query().map_or("/", |pq| pq.as_str());
    let request_uri: http::Uri = path_and_query
        .parse()
        .map_err(|e| -> FilterError { format!("http_callout: bad path in URL: {e}").into() })?;

    Ok((host, port, tls, sni, authority_str, request_uri))
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

    SubRequestClient::new(connector)
}

// -----------------------------------------------------------------------------
// HttpFilter Implementation
// -----------------------------------------------------------------------------

#[async_trait]
impl HttpFilter for HttpCalloutFilter {
    fn name(&self) -> &'static str {
        FILTER_NAME
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

        self.execute_callout(ctx, None).await
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
        self.execute_callout(ctx, body_bytes).await
    }
}
