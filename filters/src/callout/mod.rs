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

use async_trait::async_trait;
use bytes::Bytes;
use config::{FailureModeConfig, HttpCalloutConfig, Phase, expand_env_vars, validate_callout_url};
use extract::{BodyShaper, CompiledExtraction};
use praxis_core::callout::{
    CalloutClient, CalloutConfig, CalloutRequest, CalloutResponse, CalloutResult,
    CircuitBreakerConfig as CoreCircuitBreakerConfig, DEPTH_HEADER, FailureMode,
};
use praxis_filter::{
    BodyAccess, BodyMode, FilterAction, FilterError, FilterResultSet, HttpFilter, HttpFilterContext, Rejection,
    parse_filter_config,
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

    /// Reusable HTTP callout client.
    client: CalloutClient,

    /// Pre-compiled `JSONPath` extraction rules.
    extractions: Vec<CompiledExtraction>,

    /// Downstream headers to copy into the callout request.
    forward_headers: Vec<http::HeaderName>,

    /// Static headers to send with every callout.
    headers: Vec<(http::HeaderName, http::HeaderValue)>,

    /// Callout response headers to inject into the upstream
    /// request on success.
    inject_headers: Vec<http::HeaderName>,

    /// Maximum request body bytes to buffer.
    max_body_bytes: usize,

    /// When the callout fires.
    phase: Phase,

    /// Target URL for the callout.
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

        let client = build_callout_client(&cfg)?;

        Ok(Box::new(Self {
            body_shaper,
            client,
            extractions,
            forward_headers,
            headers,
            inject_headers,
            max_body_bytes: cfg.request.max_body_bytes,
            phase: cfg.request.phase,
            url: cfg.target.url,
        }))
    }

    /// Build a [`CalloutRequest`] from the current filter context.
    fn build_request(&self, ctx: &HttpFilterContext<'_>, body: Option<Vec<u8>>) -> CalloutRequest {
        let depth = ctx
            .request
            .headers
            .get(DEPTH_HEADER)
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse::<u32>().ok())
            .unwrap_or(0);

        let mut headers = self.headers.clone();

        for name in &self.forward_headers {
            if let Some(value) = ctx.request.headers.get(name) {
                headers.push((name.clone(), value.clone()));
            }
        }

        CalloutRequest {
            body,
            depth,
            headers,
            method: http::Method::POST,
            url: self.url.clone(),
        }
    }

    /// Process a successful callout response: extract results and
    /// inject headers.
    fn handle_success(
        &self,
        response: &CalloutResponse,
        ctx: &mut HttpFilterContext<'_>,
    ) -> Result<FilterAction, FilterError> {
        if !self.extractions.is_empty() {
            if let Ok(json) = serde_json::from_slice::<serde_json::Value>(&response.body) {
                let mut results = FilterResultSet::new();
                for extraction in &self.extractions {
                    extraction.evaluate(&json, &mut results)?;
                }
                debug!(results = ?results, "extracted callout results");
                match self.phase {
                    // Headers phase already runs inside `on_request`, so
                    // results are published directly for branch evaluation.
                    Phase::RequestHeaders => {
                        ctx.filter_results.insert(self.name(), results);
                    },
                    // Body phase runs during StreamBuffer pre-read, before the
                    // headers-phase pipeline. Branch evaluation clears
                    // `filter_results` after every filter, so results written
                    // now would be wiped by any preceding filter before this
                    // filter's own branches are evaluated. Stash them and
                    // re-publish in `on_request`.
                    Phase::RequestBody => ctx.insert_filter_state(results),
                }
            } else {
                warn!("callout response body is not valid JSON; skipping extraction");
            }
        }

        for name in &self.inject_headers {
            if let Some((_, value)) = response.headers.iter().find(|(n, _)| n == name)
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

    /// Map a [`CalloutResult`] to a [`FilterAction`], logging the
    /// outcome.
    fn handle_result(
        &self,
        result: CalloutResult,
        ctx: &mut HttpFilterContext<'_>,
    ) -> Result<FilterAction, FilterError> {
        match result {
            CalloutResult::Success(response) => {
                info!(url = %self.url, status = response.status, "callout succeeded");
                self.handle_success(&response, ctx)
            },
            CalloutResult::Failed => {
                warn!(url = %self.url, "callout failed; continuing (fail-open)");
                Ok(FilterAction::Continue)
            },
            CalloutResult::Rejected(rejection) => {
                info!(
                    url = %self.url, status = rejection.status,
                    "callout rejected request"
                );
                Ok(Self::build_rejection(rejection.status))
            },
        }
    }

    /// Execute the callout and process the result.
    async fn execute_callout(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        body: Option<Vec<u8>>,
    ) -> Result<FilterAction, FilterError> {
        let body_len = body.as_ref().map_or(0, Vec::len);
        let callout_body = self.shape_body(body);

        debug!(url = %self.url, body_bytes = body_len, "executing callout");

        let request = self.build_request(ctx, callout_body);
        let result = self.client.execute(request).await;
        self.handle_result(result, ctx)
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

/// Build the [`CalloutClient`] from parsed config.
#[expect(
    clippy::cast_possible_truncation,
    reason = "durations are bounded by config validation"
)]
fn build_callout_client(cfg: &HttpCalloutConfig) -> Result<CalloutClient, FilterError> {
    let failure_mode = match cfg.on_failure {
        FailureModeConfig::Closed => FailureMode::Closed,
        FailureModeConfig::Open => FailureMode::Open,
    };

    let circuit_breaker = cfg.circuit_breaker.as_ref().map(|cb| CoreCircuitBreakerConfig {
        consecutive_failures: cb.failure_threshold,
        recovery_window_ms: cb.recovery_timeout.as_millis() as u64,
    });

    let callout_config = CalloutConfig {
        circuit_breaker,
        failure_mode,
        max_depth: cfg.max_depth.unwrap_or(1),
        status_on_error: cfg.status_on_error.unwrap_or(403),
        timeout_ms: cfg.target.timeout.as_millis() as u64,
        ..CalloutConfig::default()
    };

    CalloutClient::new(callout_config).map_err(|e| -> FilterError { format!("http_callout: {e}").into() })
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
        match self.phase {
            Phase::RequestHeaders => self.execute_callout(ctx, None).await,
            Phase::RequestBody => {
                if let Some(results) = ctx.remove_filter_state::<FilterResultSet>() {
                    debug!(results = ?results, "publishing stashed callout results");
                    ctx.filter_results.insert(self.name(), results);
                }
                Ok(FilterAction::Continue)
            },
        }
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
