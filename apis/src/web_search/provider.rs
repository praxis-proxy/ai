// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Search provider abstraction and implementations.
//!
//! Uses [`SubRequestClient`] from praxis-core for HTTP callouts
//! with connection pooling, admission control, and TLS.
//!
//! [`SubRequestClient`]: praxis_core::subrequest::SubRequestClient

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use bytes::Bytes;
use http::HeaderMap;
use praxis_core::connectivity::{PreparedSubrequest, PreparedTarget, prepare_url_target};
use praxis_filter::{
    CalloutOutcome, CalloutResponse, DeferredCredential, FilterError, FilterPipeline, FilteredSubrequestExecutor,
    HttpFilterContext, IterationState, PendingCredentials, RequestExtensions, StagedUpstream, StagedUpstreamFallback,
    SubrequestRuntime,
};
use secrecy::{ExposeSecret as _, SecretString};
use serde_json::Value;
use tracing::{debug, warn};

use super::{
    ValidatedConfig,
    config::{SearchContextSize, SearchProvider},
};
use crate::callout_identity::CalloutIdentity;
use crate::subrequest::{SubRequest, SubRequestClient, SubResponse};

/// Response body cap for search callouts (1 MiB). Distinct from
/// `max_body_bytes` which governs inbound request buffering.
const MAX_SEARCH_RESPONSE_BYTES: usize = 1_048_576;

// -----------------------------------------------------------------------------
// SearchResult
// -----------------------------------------------------------------------------

/// A single search result.
#[derive(Debug, Clone)]
pub(crate) struct SearchResult {
    /// Result title.
    pub title: String,
    /// Result URL.
    pub url: String,
    /// Snippet or description.
    pub snippet: String,
}

// -----------------------------------------------------------------------------
// SearchOutcome
// -----------------------------------------------------------------------------

/// Outcome of a search execution.
#[derive(Debug)]
pub(crate) enum SearchOutcome {
    /// Search succeeded. An empty vector is a successful zero-result search.
    Results(Vec<SearchResult>),
    /// Search failed — timeout, transport error, non-2xx status, oversized
    /// response, or unparseable body. Callers continue with a truthful failed
    /// tool result rather than exposing provider details to the client.
    Failed,
}

// -----------------------------------------------------------------------------
// CalloutContext
// -----------------------------------------------------------------------------

/// Downstream caller attributes and outbound recursion depth captured for one
/// web-search callout.
///
/// The owning filter builds this from its [`HttpFilterContext`] so the outbound
/// chain and the [`FilteredSubrequestExecutor`] observe the *originating*
/// request rather than a synthetic anonymous caller at depth zero:
///
/// - `runtime` forwards the client address, downstream TLS state, peer identity, and request-start instant, so outbound
///   security and observability filters see the real client and duration accounting stays consistent.
/// - `depth` carries the request's current outbound recursion depth (read from the IRR-owned [`IterationState`] in
///   request extensions), so the executor stamps the callout at `depth + 1` and a recursive target remains accountable
///   to the IRR depth ceiling instead of resetting the count.
///
/// [`FilteredSubrequestExecutor`]: praxis_filter::FilteredSubrequestExecutor
pub(crate) struct CalloutContext {
    /// Downstream attributes forwarded to the outbound chain.
    runtime: SubrequestRuntime,
    /// Current outbound recursion depth of the owning request.
    depth: u8,
}

impl CalloutContext {
    /// Capture the caller context and current outbound depth from the owning
    /// filter's request context.
    ///
    /// The `peer_identity` `Arc` is cloned at this ownership boundary because
    /// [`SubrequestRuntime`] owns the forwarded identity for the callout's
    /// lifetime; the clone only bumps a refcount.
    pub(crate) fn from_filter_context(ctx: &HttpFilterContext<'_>) -> Self {
        Self {
            runtime: SubrequestRuntime::new(
                ctx.client_addr,
                ctx.downstream_tls,
                ctx.peer_identity.clone(),
                ctx.request_start,
            ),
            depth: current_outbound_depth(ctx),
        }
    }

    /// Build a synthetic top-level callout context for tests: an anonymous
    /// caller (no client address, plaintext downstream, no peer identity) at
    /// outbound depth zero.
    #[cfg(test)]
    pub(crate) fn for_test() -> Self {
        Self {
            runtime: SubrequestRuntime::new(None, false, None, Instant::now()),
            depth: 0,
        }
    }
}

/// Read the request's current outbound recursion depth from the IRR-owned
/// [`IterationState`] in request extensions.
///
/// The `iterative_request_router` captures the inbound `x-praxis-iterative-depth`
/// framework header into [`IterationState`] and then strips every reserved
/// `x-praxis-*` header before it builds each step's context, so a step filter can
/// no longer read the depth from the request headers — it must read the state the
/// router injected. A request with no [`IterationState`] (a top-level, non-IRR
/// placement such as the standalone `web-search.yaml`) is at depth zero.
fn current_outbound_depth(ctx: &HttpFilterContext<'_>) -> u8 {
    ctx.extensions.get::<IterationState>().map_or(0, IterationState::depth)
}

// -----------------------------------------------------------------------------
// SearchClient
// -----------------------------------------------------------------------------

/// HTTP search client using [`SubRequestClient`] from praxis-core.
///
/// [`SubRequestClient`]: praxis_core::subrequest::SubRequestClient
pub(crate) struct SearchClient {
    /// Sub-request client for bounded search callouts.
    client: SubRequestClient,
    /// Per-request timeout.
    timeout: Duration,
    /// Search backend provider.
    provider: SearchProvider,
    /// API key for the search provider.
    api_key: SecretString,
    /// Default search context size.
    default_context_size: SearchContextSize,
    /// Override the provider's default API base URL.
    base_url: Option<String>,
}

impl std::fmt::Debug for SearchClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SearchClient")
            .field("client", &self.client)
            .field("timeout", &self.timeout)
            .field("provider", &self.provider)
            .field("api_key", &"[REDACTED]")
            .field("default_context_size", &self.default_context_size)
            .field("base_url", &self.base_url)
            .finish()
    }
}

impl SearchClient {
    /// Build a search client from validated filter config.
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] if the API key header value is
    /// not valid ASCII.
    pub(crate) fn from_config(
        filter_name: &'static str,
        config: &ValidatedConfig,
        subrequest_client: SubRequestClient,
    ) -> Result<Self, FilterError> {
        http::HeaderValue::from_str(config.api_key.expose_secret())
            .map_err(|e| FilterError::from(format!("{filter_name}: invalid API key header value: {e}")))?;
        Ok(Self {
            client: subrequest_client,
            timeout: Duration::from_millis(config.timeout_ms),
            provider: config.provider,
            api_key: config.api_key.clone(),
            default_context_size: config.default_context_size,
            base_url: config.base_url.clone(),
        })
    }

    /// Execute a web search query through the bound outbound filter chain.
    ///
    /// `outbound` is the provider callout's prebuilt outbound chain (bound once
    /// at pipeline-build time via `ChainBindingContext::bind_chain`). The chain
    /// carries cross-cutting concerns (observability, security, credential
    /// injection); destination authority, DNS/SSRF, TLS/SNI, and `Host` are
    /// enforced centrally by the executor at transport time.
    pub(crate) async fn search(
        &self,
        outbound: &Arc<FilterPipeline>,
        callout: CalloutContext,
        query: &str,
        context_size: Option<SearchContextSize>,
        identity: &CalloutIdentity,
    ) -> SearchOutcome {
        let size = context_size.unwrap_or(self.default_context_size);
        let count = size.result_count();
        debug!(
            provider = self.provider.as_str(),
            query_bytes = query.len(),
            count,
            "executing web search"
        );
        let (url, request) = match self.provider {
            SearchProvider::Brave => self.build_brave_request(query, count),
            SearchProvider::Tavily => self.build_tavily_request(query, size),
            SearchProvider::You => self.build_you_request(query, count),
        };
        self.execute_search(outbound, callout, &url, request, identity).await
    }

    /// Execute a search request through the outbound chain and map the response
    /// to a [`SearchOutcome`].
    ///
    /// The provider destination is resolved with [`prepare_url_target`] (a
    /// permissive validation hook — private-address/SSRF enforcement is deferred
    /// to the executor's connect-time `build_peer`, gated by
    /// `insecure_options.allow_private_upstreams`) and staged as a
    /// [`StagedUpstream`] (plus a [`StagedUpstreamFallback`] address set) in the
    /// sub-request extensions. The executor seeds `filter_ctx.upstream` from the
    /// staged upstream before the request phase and re-pins it afterward, so the
    /// outbound chain needs no upstream-selecting filter and cannot retarget the
    /// callout. [`PreparedTarget::bind`] rewrites the request `Host` to the URL
    /// authority and its target to the origin-form path+query.
    ///
    /// [`PreparedTarget::bind`]: praxis_core::connectivity::PreparedTarget::bind
    async fn execute_search(
        &self,
        outbound: &Arc<FilterPipeline>,
        callout: CalloutContext,
        url: &str,
        request: SubRequest,
        identity: &CalloutIdentity,
    ) -> SearchOutcome {
        let deadline = Instant::now() + self.timeout;
        let Some((prepared, extensions)) = self.prepare_staged_request(url, request, deadline, identity).await else {
            return SearchOutcome::Failed;
        };
        // Build the executor here (a small struct) so the caller context is
        // consumed at the staging boundary; the large `run` future stays boxed
        // inside `dispatch_callout`.
        let executor = FilteredSubrequestExecutor::for_callout(
            self.client.clone(),
            callout.runtime,
            callout.depth,
            MAX_SEARCH_RESPONSE_BYTES,
            self.timeout,
        );
        let result = Self::dispatch_callout(executor, outbound, prepared.request(), extensions, deadline).await;
        self.map_callout_outcome(result)
    }

    /// Resolve and stage the provider destination for a callout.
    ///
    /// Returns the bound request and the executor extensions: the pinned callout
    /// [`StagedUpstream`], its [`StagedUpstreamFallback`] address set, and — for a
    /// header-authenticated provider — the API key staged as an authority-bound
    /// [`PendingCredentials`]. A resolution or staging failure logs and returns
    /// `None`, which the caller maps to [`SearchOutcome::Failed`].
    async fn prepare_staged_request(
        &self,
        url: &str,
        request: SubRequest,
        deadline: Instant,
        identity: &CalloutIdentity,
    ) -> Option<(PreparedSubrequest, RequestExtensions)> {
        // Private-address / SSRF enforcement is deferred to the executor's
        // connect-time `build_peer`, so this hook is permissive.
        let target = match prepare_url_target(url, deadline, |_addrs| Ok(())).await {
            Ok(target) => target,
            Err(e) => {
                warn!(provider = self.provider.as_str(), error = %e, "search callout target preparation failed");
                return None;
            },
        };
        // Assemble the executor extensions while `target` is still owned, then
        // let `bind` consume it into the origin-form request.
        let extensions = self.stage_extensions(&target, url, identity)?;
        Some((target.bind(request), extensions))
    }

    /// Assemble the executor extensions for a resolved target: the pinned callout
    /// [`StagedUpstream`], its [`StagedUpstreamFallback`] address set, the caller's
    /// trusted [`StateOwner`] when present, and — for a header-authenticated provider
    /// — the API key staged as an authority-bound [`PendingCredentials`]. Any staging
    /// failure logs and returns `None` (fail-closed) rather than dialing the provider
    /// degraded.
    ///
    /// [`StateOwner`]: crate::StateOwner
    fn stage_extensions(&self, target: &PreparedTarget, url: &str, identity: &CalloutIdentity) -> Option<RequestExtensions> {
        // Stage the resolved upstream (pinned to the primary validated address)
        // and the full validated address set. The executor seeds
        // `filter_ctx.upstream` from the `StagedUpstream` before the request
        // phase, re-pins it afterward to defeat mid-chain retargeting, and — on a
        // connection refusal to the pinned address — advances through the
        // `StagedUpstreamFallback` addresses (the DNS fallback a direct-dial
        // transport performs) without re-resolving. Every address was
        // SSRF-validated together and each literal is re-checked at connect time.
        let staged = match StagedUpstream::from_prepared_target(target) {
            Ok(staged) => staged,
            Err(e) => {
                warn!(provider = self.provider.as_str(), error = %e, "search callout upstream staging failed");
                return None;
            },
        };
        let fallback = StagedUpstreamFallback::from_prepared_target(target);
        // Stage the provider's static API credential so the executor injects it
        // only after it has resolved and pinned the destination. A staging
        // failure for a header-authenticated provider fails the callout closed
        // rather than dialing the provider unauthenticated.
        let pending = match self.staged_credentials(url, identity) {
            Ok(pending) => pending,
            Err(e) => {
                warn!(provider = self.provider.as_str(), error = %e, "search callout credential staging failed");
                return None;
            },
        };
        let mut extensions = RequestExtensions::default();
        extensions.insert(staged);
        extensions.insert(fallback);
        if let Some(pending) = pending {
            extensions.insert(pending);
        }
        // Project the caller's trusted owner into the isolated child subrequest so
        // the outbound chain (e.g. `state_owner_headers`) can attribute the callout
        // to the real caller. Mirrors `crate::state_owner::project_state_owner`; the
        // bounded three-string clone is the sanctioned ownership boundary.
        if let Some(owner) = identity.owner.as_ref() {
            extensions.insert(owner.clone());
        }
        Some(extensions)
    }

    /// The request header a header-authenticated provider carries its API key in,
    /// or `None` for a body-authenticated provider (Tavily, whose key travels in
    /// the request body and is protected instead by the executor re-pinning the
    /// staged upstream against mid-chain retargeting).
    fn auth_header(&self) -> Option<http::HeaderName> {
        match self.provider {
            SearchProvider::Brave => Some(http::HeaderName::from_static("x-subscription-token")),
            SearchProvider::You => Some(http::HeaderName::from_static("x-api-key")),
            SearchProvider::Tavily => None,
        }
    }

    /// Stage a header-authenticated provider's API key as an authority-bound
    /// [`DeferredCredential`].
    ///
    /// Deferring the secret keeps it out of the outbound chain entirely: the
    /// executor injects it only after it resolves the destination and only into a
    /// request bound for the URL's host, so a chain filter can neither observe the
    /// secret (it is never present on the in-chain request) nor exfiltrate it by
    /// retargeting `ctx.upstream` to another authority (the credential is bound to
    /// the provider host and is dropped, zeroized, on any authority mismatch). The
    /// credential is bound host-wildcard so it matches the executor's resolved
    /// destination whether the URL used the provider default port or a `base_url`
    /// override with an explicit port. The secret is the caller's per-user credential
    /// when present, else the shared provider key.
    ///
    /// Returns `Ok(None)` for a body-authenticated provider (Tavily) and
    /// `Err(_)` if the URL has no host or the key is not a valid header value.
    fn staged_credentials(&self, url: &str, identity: &CalloutIdentity) -> Result<Option<PendingCredentials>, FilterError> {
        let Some(header) = self.auth_header() else {
            return Ok(None);
        };
        let host = http::Uri::try_from(url)
            .ok()
            .and_then(|uri| uri.host().map(str::to_owned))
            .ok_or_else(|| FilterError::from("search callout URL has no host for credential binding".to_owned()))?;
        // Prefer the caller's per-user secret from the configured slot; fall back to
        // the shared provider key. Both are deferred (never placed on the in-chain
        // request) and injected by the executor only at the resolved, pinned host.
        let secret = identity
            .user_credential
            .as_ref()
            .map_or_else(|| self.api_key.expose_secret(), |user| user.expose_secret());
        let credential = DeferredCredential::new_host_wildcard(&host, header, secret)?;
        let mut pending = PendingCredentials::new();
        pending.push(credential);
        Ok(Some(pending))
    }

    /// Run the prepared callout through the outbound chain executor.
    ///
    /// Kept as a dedicated leaf so the executor's large run future is boxed in a
    /// minimal frame: the box keeps this future's state machine small (satisfies
    /// `large_futures`) while the surrounding locals stay off the stack that
    /// materializes the box (satisfies `large_stack_frames`).
    async fn dispatch_callout(
        executor: FilteredSubrequestExecutor,
        outbound: &Arc<FilterPipeline>,
        request: &SubRequest,
        extensions: RequestExtensions,
        deadline: Instant,
    ) -> Result<CalloutOutcome, FilterError> {
        Box::pin(executor.run_classified(outbound, request, extensions, deadline)).await
    }

    /// Map the executor's classified callout outcome to a [`SearchOutcome`].
    ///
    /// A streaming response is a misconfiguration for a bounded search callout;
    /// an oversized response and any error fail closed; provider specifics never
    /// reach the client. Unlike a callout whose oversized response maps to a
    /// client HTTP 413, a web-search callout feeds a tool result into the agentic
    /// loop and never surfaces a provider-response status to the client, so
    /// [`CalloutOutcome::ResponseTooLarge`] fails the callout closed — the model
    /// then continues with a truthful "search unavailable" result.
    fn map_callout_outcome(&self, result: Result<CalloutOutcome, FilterError>) -> SearchOutcome {
        match result {
            Ok(outcome) => self.map_callout_success(outcome),
            Err(e) => {
                warn!(provider = self.provider.as_str(), error = %e, "search callout failed");
                SearchOutcome::Failed
            },
        }
    }

    /// Map a successfully executed callout's classified outcome to a
    /// [`SearchOutcome`], failing closed on anything but a bounded buffered
    /// response. Split from [`map_callout_outcome`](Self::map_callout_outcome)
    /// so each stays within the cognitive-complexity budget.
    fn map_callout_success(&self, outcome: CalloutOutcome) -> SearchOutcome {
        match outcome {
            CalloutOutcome::Response(CalloutResponse::Buffered(response)) => self.map_search_result(&response),
            CalloutOutcome::Response(CalloutResponse::Streaming { .. }) => {
                warn!(
                    provider = self.provider.as_str(),
                    "search callout produced a streaming response; treating as failed"
                );
                SearchOutcome::Failed
            },
            CalloutOutcome::ResponseTooLarge { actual, limit } => {
                warn!(
                    provider = self.provider.as_str(),
                    ?actual,
                    limit,
                    "search callout response exceeded the size limit; treating as failed"
                );
                SearchOutcome::Failed
            },
            // `CalloutOutcome` is `#[non_exhaustive]`: a future classified outcome
            // fails the callout closed rather than being read as a success.
            _ => {
                warn!(
                    provider = self.provider.as_str(),
                    "search callout produced an unrecognized outcome; treating as failed"
                );
                SearchOutcome::Failed
            },
        }
    }

    /// Map a buffered sub-request response to a [`SearchOutcome`].
    ///
    /// Non-2xx statuses map to [`SearchOutcome::Failed`]. Detailed diagnostics
    /// are logged; provider specifics never reach the client.
    fn map_search_result(&self, response: &SubResponse) -> SearchOutcome {
        if (200..300).contains(&(response.status as usize)) {
            self.parse_response(&response.body)
        } else {
            warn!(
                provider = self.provider.as_str(),
                status = response.status,
                "search callout returned non-2xx"
            );
            SearchOutcome::Failed
        }
    }

    /// Build a Brave Search API request.
    fn build_brave_request(&self, query: &str, count: u32) -> (String, SubRequest) {
        let encoded_query = percent_encoding::utf8_percent_encode(query, percent_encoding::NON_ALPHANUMERIC);
        let base = self.base_url.as_deref().unwrap_or("https://api.search.brave.com");
        let url = format!("{base}/res/v1/web/search?q={encoded_query}&count={count}");

        // The API key is NOT set here: it is staged as an authority-bound
        // `DeferredCredential` (`x-subscription-token`) in `prepare_staged_request`
        // and injected by the executor only after it resolves and pins the
        // destination, so the outbound chain never observes the secret and it can
        // only reach the provider host it was prepared for.
        let mut headers = HeaderMap::new();
        headers.insert(http::header::ACCEPT, http::HeaderValue::from_static("application/json"));

        (
            url,
            SubRequest {
                method: http::Method::GET,
                uri: "/".parse().unwrap_or_default(),
                headers,
                body: Bytes::new(),
            },
        )
    }

    /// Build a Tavily Search API request.
    fn build_tavily_request(&self, query: &str, context_size: SearchContextSize) -> (String, SubRequest) {
        let search_depth = match context_size {
            SearchContextSize::Low | SearchContextSize::Medium => "basic",
            SearchContextSize::High => "advanced",
        };
        let max_results = context_size.result_count();

        let body = serde_json::json!({
            "api_key": self.api_key.expose_secret(),
            "query": query,
            "search_depth": search_depth,
            "max_results": max_results,
        });

        let mut headers = HeaderMap::new();
        headers.insert(
            http::header::CONTENT_TYPE,
            http::HeaderValue::from_static("application/json"),
        );
        headers.insert(http::header::ACCEPT, http::HeaderValue::from_static("application/json"));

        let base = self.base_url.as_deref().unwrap_or("https://api.tavily.com");
        let url = format!("{base}/search");
        (
            url,
            SubRequest {
                method: http::Method::POST,
                uri: "/".parse().unwrap_or_default(),
                headers,
                body: Bytes::from(serde_json::to_vec(&body).unwrap_or_default()),
            },
        )
    }

    /// Build a You.com Search API request.
    fn build_you_request(&self, query: &str, count: u32) -> (String, SubRequest) {
        let body = serde_json::json!({
            "query": query,
            "count": count,
        });

        // The API key is NOT set here: it is staged as an authority-bound
        // `DeferredCredential` (`x-api-key`) in `prepare_staged_request` and
        // injected by the executor only after it resolves and pins the
        // destination, so the outbound chain never observes the secret and it can
        // only reach the provider host it was prepared for.
        let mut headers = HeaderMap::new();
        headers.insert(
            http::header::CONTENT_TYPE,
            http::HeaderValue::from_static("application/json"),
        );
        headers.insert(http::header::ACCEPT, http::HeaderValue::from_static("application/json"));

        let base = self.base_url.as_deref().unwrap_or("https://api.you.com");
        let url = format!("{base}/v1/search");
        (
            url,
            SubRequest {
                method: http::Method::POST,
                uri: "/".parse().unwrap_or_default(),
                headers,
                body: Bytes::from(serde_json::to_vec(&body).unwrap_or_default()),
            },
        )
    }

    /// Parse search results from the provider's JSON response.
    fn parse_response(&self, body: &[u8]) -> SearchOutcome {
        let json: Value = match serde_json::from_slice(body) {
            Ok(v) => v,
            Err(e) => {
                warn!(provider = self.provider.as_str(), error = %e, "failed to parse search response");
                return SearchOutcome::Failed;
            },
        };

        let results = match self.provider {
            SearchProvider::Brave => parse_brave_results(&json),
            SearchProvider::Tavily => parse_tavily_results(&json),
            SearchProvider::You => parse_you_results(&json),
        };

        debug!(
            provider = self.provider.as_str(),
            count = results.len(),
            "parsed search results"
        );

        SearchOutcome::Results(results)
    }
}

// -----------------------------------------------------------------------------
// Provider-specific parsers
// -----------------------------------------------------------------------------

/// Parse Brave Search API response.
///
/// Expected shape: `{ "web": { "results": [ { "title", "url", "description" } ] } }`
fn parse_brave_results(json: &Value) -> Vec<SearchResult> {
    json.get("web")
        .and_then(|web| web.get("results"))
        .and_then(Value::as_array)
        .map(|results| {
            results
                .iter()
                .filter_map(|r| {
                    Some(SearchResult {
                        title: r.get("title")?.as_str()?.to_owned(),
                        url: r.get("url")?.as_str()?.to_owned(),
                        snippet: r
                            .get("description")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_owned(),
                    })
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Parse Tavily Search API response.
///
/// Expected shape: `{ "results": [ { "title", "url", "content" } ] }`
fn parse_tavily_results(json: &Value) -> Vec<SearchResult> {
    json.get("results")
        .and_then(Value::as_array)
        .map(|results| {
            results
                .iter()
                .filter_map(|r| {
                    Some(SearchResult {
                        title: r.get("title")?.as_str()?.to_owned(),
                        url: r.get("url")?.as_str()?.to_owned(),
                        snippet: r.get("content").and_then(Value::as_str).unwrap_or_default().to_owned(),
                    })
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Parse You.com Search API results.
///
/// Expected shape: `{ "results": { "web": [ { "title", "url", "description" } ], "news": [...] } }`.
fn parse_you_results(json: &Value) -> Vec<SearchResult> {
    ["web", "news"]
        .into_iter()
        .filter_map(|section| json.get("results")?.get(section)?.as_array())
        .flatten()
        .filter_map(|result| {
            Some(SearchResult {
                title: result.get("title")?.as_str()?.to_owned(),
                url: result.get("url")?.as_str()?.to_owned(),
                snippet: result
                    .get("description")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned(),
            })
        })
        .collect()
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    reason = "tests"
)]
mod tests {
    use std::{
        io::{Read as _, Write as _},
        net::TcpListener,
    };

    use praxis_core::connectivity::Upstream;
    use secrecy::SecretString;
    use serde_json::json;

    use super::*;

    fn test_subrequest_client() -> SubRequestClient {
        SubRequestClient::new(praxis_core::subrequest::SubRequestConnector::new(4, None))
    }

    #[test]
    fn parse_brave_results_normal() {
        let json = json!({
            "web": {
                "results": [
                    {"title": "Rust Lang", "url": "https://rust-lang.org", "description": "Systems programming"},
                    {"title": "Crates.io", "url": "https://crates.io", "description": "Rust packages"}
                ]
            }
        });
        let results = parse_brave_results(&json);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].title, "Rust Lang");
        assert_eq!(results[0].url, "https://rust-lang.org");
        assert_eq!(results[0].snippet, "Systems programming");
        assert_eq!(results[1].title, "Crates.io");
    }

    #[test]
    fn parse_brave_results_empty() {
        let json = json!({"web": {"results": []}});
        assert!(
            parse_brave_results(&json).is_empty(),
            "empty Brave results should parse as empty"
        );
    }

    #[test]
    fn parse_brave_results_missing_web() {
        let json = json!({"query": "test"});
        assert!(
            parse_brave_results(&json).is_empty(),
            "missing Brave web results should parse as empty"
        );
    }

    #[test]
    fn parse_brave_results_skips_incomplete() {
        let json = json!({
            "web": {
                "results": [
                    {"title": "Good", "url": "https://example.com", "description": "ok"},
                    {"description": "missing title and url"},
                    {"title": "Also Good", "url": "https://example.org"}
                ]
            }
        });
        let results = parse_brave_results(&json);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn parse_tavily_results_normal() {
        let json = json!({
            "results": [
                {"title": "Example", "url": "https://example.com", "content": "Description here"},
                {"title": "Another", "url": "https://another.com", "content": "More info"}
            ]
        });
        let results = parse_tavily_results(&json);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].title, "Example");
        assert_eq!(results[0].snippet, "Description here");
    }

    #[test]
    fn parse_tavily_results_empty() {
        let json = json!({"results": []});
        assert!(
            parse_tavily_results(&json).is_empty(),
            "empty Tavily results should parse as empty"
        );
    }

    #[test]
    fn parse_tavily_results_missing_results() {
        let json = json!({"answer": "some answer"});
        assert!(
            parse_tavily_results(&json).is_empty(),
            "missing Tavily results should parse as empty"
        );
    }

    #[test]
    fn parse_tavily_results_missing_content() {
        let json = json!({
            "results": [
                {"title": "No Content", "url": "https://example.com"}
            ]
        });
        let results = parse_tavily_results(&json);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].snippet, "");
    }

    #[test]
    fn build_you_request_defers_api_key_and_sends_body() {
        let config = ValidatedConfig {
            provider: SearchProvider::You,
            api_key: SecretString::from("test-key".to_owned()),
            default_context_size: SearchContextSize::Medium,
            timeout_ms: 5000,
            max_body_bytes: 64 * 1024 * 1024,
            base_url: None,
            user_credential: None,
            terminal_streaming: false,
        };
        let client = SearchClient::from_config("test", &config, test_subrequest_client()).unwrap();

        let (url, request) = client.build_you_request("Praxis proxy", 5);

        assert_eq!(url, "https://api.you.com/v1/search");
        assert_eq!(request.method, http::Method::POST);
        // The API key must NOT ride on the in-chain request: it is deferred and
        // injected by the executor only after the destination is resolved and
        // pinned, so the outbound chain never observes the secret.
        assert!(
            request.headers.get("x-api-key").is_none(),
            "You.com API key must be deferred, not set on the in-chain request"
        );
        assert_eq!(
            serde_json::from_slice::<Value>(&request.body).unwrap(),
            json!({"query": "Praxis proxy", "count": 5})
        );

        // The deferred credential is staged bound to the provider host so the
        // executor injects it at transport time.
        let pending = client
            .staged_credentials(&url, &shared_key_identity())
            .expect("staging a valid You.com credential must succeed")
            .expect("You.com authenticates via a header credential");
        assert!(!pending.is_empty(), "You.com must stage a deferred credential");
    }

    fn test_client_for(provider: SearchProvider) -> SearchClient {
        let config = ValidatedConfig {
            provider,
            api_key: SecretString::from("test-key".to_owned()),
            default_context_size: SearchContextSize::Medium,
            timeout_ms: 5000,
            max_body_bytes: 64 * 1024 * 1024,
            base_url: None,
            user_credential: None,
            terminal_streaming: false,
        };
        SearchClient::from_config("test", &config, test_subrequest_client()).unwrap()
    }

    /// A callout identity with no owner and no per-user credential (shared-key path).
    fn shared_key_identity() -> CalloutIdentity {
        CalloutIdentity { owner: None, user_credential: None }
    }

    #[test]
    fn brave_defers_credential_off_the_in_chain_request() {
        let brave = test_client_for(SearchProvider::Brave);
        let (url, request) = brave.build_brave_request("test", 5);
        assert!(
            request.headers.get("x-subscription-token").is_none(),
            "Brave API key must be deferred, not set on the in-chain request"
        );
        assert!(
            brave
                .staged_credentials(&url, &shared_key_identity())
                .expect("staging a valid Brave credential must succeed")
                .is_some_and(|pending| !pending.is_empty()),
            "Brave must stage a deferred credential"
        );
    }

    #[test]
    fn tavily_stages_no_header_credential() {
        // Tavily carries its key in the body, so it stages no header credential.
        let tavily = test_client_for(SearchProvider::Tavily);
        let (url, _) = tavily.build_tavily_request("test", SearchContextSize::Medium);
        assert!(
            tavily
                .staged_credentials(&url, &shared_key_identity())
                .expect("Tavily credential staging must not error")
                .is_none(),
            "Tavily authenticates via the body and must not stage a header credential"
        );
    }

    #[test]
    fn parse_you_results_merges_web_and_news_sections() {
        let json = json!({
            "results": {
                "web": [
                    {"title": "Praxis", "url": "https://praxis.example", "description": "Proxy"},
                    {"description": "Missing identity"}
                ],
                "news": [
                    {"title": "vLLM", "url": "https://vllm.example", "description": "Inference"}
                ]
            }
        });

        let results = parse_you_results(&json);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].title, "Praxis");
        assert_eq!(results[0].snippet, "Proxy");
        assert_eq!(results[1].title, "vLLM");
    }

    #[test]
    fn parse_you_results_handles_missing_sections() {
        assert!(
            parse_you_results(&json!({"results": {}})).is_empty(),
            "missing You.com result sections should parse as empty"
        );
    }

    #[test]
    fn search_client_from_config() {
        let config = ValidatedConfig {
            provider: SearchProvider::Brave,
            api_key: SecretString::from("test-key".to_owned()),
            default_context_size: SearchContextSize::Medium,
            timeout_ms: 5000,
            max_body_bytes: 64 * 1024 * 1024,
            base_url: None,
            user_credential: None,
            terminal_streaming: false,
        };
        let client = SearchClient::from_config("test", &config, test_subrequest_client());
        assert!(client.is_ok(), "a valid search configuration should build a client");
    }

    #[test]
    fn invalid_api_key_diagnostic_names_owner() {
        let config = ValidatedConfig {
            provider: SearchProvider::Brave,
            api_key: SecretString::from("invalid\nkey".to_owned()),
            default_context_size: SearchContextSize::Medium,
            timeout_ms: 5000,
            max_body_bytes: 64 * 1024 * 1024,
            base_url: None,
            user_credential: None,
            terminal_streaming: false,
        };

        let error = SearchClient::from_config("anthropic_web_search", &config, test_subrequest_client()).unwrap_err();

        assert!(
            error
                .to_string()
                .contains("anthropic_web_search: invalid API key header value"),
            "diagnostic should name the owning filter: {error}"
        );
    }

    #[test]
    fn base_url_overrides_brave_url() {
        let config = ValidatedConfig {
            provider: SearchProvider::Brave,
            api_key: SecretString::from("test-key".to_owned()),
            default_context_size: SearchContextSize::Medium,
            timeout_ms: 5000,
            max_body_bytes: 64 * 1024 * 1024,
            base_url: Some("http://localhost:9999".into()),
            user_credential: None,
            terminal_streaming: false,
        };
        let client = SearchClient::from_config("test", &config, test_subrequest_client()).unwrap();
        let (url, _) = client.build_brave_request("test query", 5);
        assert!(
            url.starts_with("http://localhost:9999/"),
            "base_url should override the default Brave URL; got: {url}"
        );
    }

    #[test]
    fn base_url_overrides_tavily_url() {
        let config = ValidatedConfig {
            provider: SearchProvider::Tavily,
            api_key: SecretString::from("test-key".to_owned()),
            default_context_size: SearchContextSize::Medium,
            timeout_ms: 5000,
            max_body_bytes: 64 * 1024 * 1024,
            base_url: Some("http://localhost:9999".into()),
            user_credential: None,
            terminal_streaming: false,
        };
        let client = SearchClient::from_config("test", &config, test_subrequest_client()).unwrap();
        let (url, _) = client.build_tavily_request("test query", SearchContextSize::Medium);
        assert!(
            url.starts_with("http://localhost:9999/"),
            "base_url should override the default Tavily URL; got: {url}"
        );
    }

    #[test]
    fn base_url_overrides_you_url() {
        let config = ValidatedConfig {
            provider: SearchProvider::You,
            api_key: SecretString::from("test-key".to_owned()),
            default_context_size: SearchContextSize::Medium,
            timeout_ms: 5000,
            max_body_bytes: 64 * 1024 * 1024,
            base_url: Some("http://localhost:9999".into()),
            user_credential: None,
            terminal_streaming: false,
        };
        let client = SearchClient::from_config("test", &config, test_subrequest_client()).unwrap();
        let (url, _) = client.build_you_request("test query", 5);
        assert!(
            url.starts_with("http://localhost:9999/"),
            "base_url should override the default You.com URL; got: {url}"
        );
    }

    #[test]
    fn parse_failure_returns_failed() {
        let config = ValidatedConfig {
            provider: SearchProvider::Brave,
            api_key: SecretString::from("test-key".to_owned()),
            default_context_size: SearchContextSize::Medium,
            timeout_ms: 5000,
            max_body_bytes: 64 * 1024 * 1024,
            base_url: None,
            user_credential: None,
            terminal_streaming: false,
        };
        let client = SearchClient::from_config("test", &config, test_subrequest_client()).unwrap();
        let outcome = client.parse_response(b"not json");
        assert!(
            matches!(outcome, SearchOutcome::Failed),
            "an unparseable 2xx body should map to Failed: {outcome:?}"
        );
    }

    #[test]
    fn parse_empty_results_is_successful_zero_result_search() {
        let config = ValidatedConfig {
            provider: SearchProvider::Brave,
            api_key: SecretString::from("test-key".to_owned()),
            default_context_size: SearchContextSize::Medium,
            timeout_ms: 5000,
            max_body_bytes: 64 * 1024 * 1024,
            base_url: None,
            user_credential: None,
            terminal_streaming: false,
        };
        let client = SearchClient::from_config("test", &config, test_subrequest_client()).unwrap();
        let outcome = client.parse_response(br#"{"web":{"results":[]}}"#);
        assert!(
            matches!(&outcome, SearchOutcome::Results(results) if results.is_empty()),
            "a parseable 2xx body with zero results is a successful empty search: {outcome:?}"
        );
    }

    fn test_search_client() -> SearchClient {
        let config = ValidatedConfig {
            provider: SearchProvider::Brave,
            api_key: SecretString::from("test-key".to_owned()),
            default_context_size: SearchContextSize::Medium,
            timeout_ms: 1000,
            max_body_bytes: 64 * 1024 * 1024,
            base_url: None,
            user_credential: None,
            terminal_streaming: false,
        };
        SearchClient::from_config("test", &config, test_subrequest_client()).unwrap()
    }

    /// Build a minimal bound outbound chain for callout tests.
    ///
    /// The chain holds an observable `request_id` builtin, standing in for the
    /// cross-cutting filters a real deployment binds; the executor seeds and
    /// re-pins the callout upstream centrally from the staged `StagedUpstream`,
    /// so no seed filter is needed. `allow_private_upstreams` is enabled because
    /// these tests dial `127.0.0.1`; production drives that flag from
    /// `insecure_options.allow_private_upstreams`.
    fn test_outbound() -> Arc<FilterPipeline> {
        Arc::new(crate::web_search::test_outbound_pipeline().unwrap())
    }

    fn spawn_http_server(listener: TcpListener, status: u16, body: &str) {
        let body = body.to_owned();
        std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut buf = [0_u8; 4096];
            let _n = stream.read(&mut buf).unwrap();
            let response = format!(
                "HTTP/1.1 {status} OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            stream.write_all(response.as_bytes()).unwrap();
        });
    }

    #[tokio::test]
    async fn search_2xx_with_valid_json_returns_results() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        spawn_http_server(
            listener,
            200,
            &json!({
                "web": {"results": [{"title": "Hit", "url": "https://hit.example", "description": "found"}]}
            })
            .to_string(),
        );

        let client = test_search_client();
        let url = format!("http://{addr}/res/v1/web/search?q=test&count=5");
        let request = SubRequest {
            method: http::Method::GET,
            uri: "/".parse().unwrap(),
            headers: HeaderMap::new(),
            body: Bytes::new(),
        };

        let outcome = client
            .execute_search(&test_outbound(), CalloutContext::for_test(), &url, request, &shared_key_identity())
            .await;
        assert!(
            matches!(&outcome, SearchOutcome::Results(r) if r.len() == 1),
            "2xx with valid JSON should return results: {outcome:?}"
        );
    }

    #[tokio::test]
    async fn search_non_2xx_returns_failed() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        spawn_http_server(listener, 500, "internal error");

        let client = test_search_client();
        let url = format!("http://{addr}/search");
        let request = SubRequest {
            method: http::Method::GET,
            uri: "/".parse().unwrap(),
            headers: HeaderMap::new(),
            body: Bytes::new(),
        };

        let outcome = client
            .execute_search(&test_outbound(), CalloutContext::for_test(), &url, request, &shared_key_identity())
            .await;
        assert!(
            matches!(outcome, SearchOutcome::Failed),
            "a non-2xx status should map to Failed: {outcome:?}"
        );
    }

    #[tokio::test]
    async fn search_connection_failure_returns_failed() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            drop(stream);
        });

        let client = test_search_client();
        let url = format!("http://{addr}/search");
        let request = SubRequest {
            method: http::Method::GET,
            uri: "/".parse().unwrap(),
            headers: HeaderMap::new(),
            body: Bytes::new(),
        };

        let outcome = client
            .execute_search(&test_outbound(), CalloutContext::for_test(), &url, request, &shared_key_identity())
            .await;
        assert!(
            matches!(outcome, SearchOutcome::Failed),
            "a transport failure should map to Failed: {outcome:?}"
        );
    }

    #[tokio::test]
    async fn search_timeout_returns_failed() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (_stream, _) = listener.accept().await.unwrap();
            tokio::time::sleep(Duration::from_secs(5)).await;
        });

        let mut client = test_search_client();
        client.timeout = Duration::from_millis(50);
        let url = format!("http://{addr}/search");
        let request = SubRequest {
            method: http::Method::GET,
            uri: "/".parse().unwrap(),
            headers: HeaderMap::new(),
            body: Bytes::new(),
        };

        let outcome = client
            .execute_search(&test_outbound(), CalloutContext::for_test(), &url, request, &shared_key_identity())
            .await;
        assert!(
            matches!(outcome, SearchOutcome::Failed),
            "a timeout should map to Failed: {outcome:?}"
        );
    }

    #[tokio::test]
    async fn search_oversized_response_returns_failed() {
        // A provider body over `MAX_SEARCH_RESPONSE_BYTES` trips the executor's
        // response ceiling, which `run_classified` surfaces as the typed
        // `CalloutOutcome::ResponseTooLarge`. A web-search callout has no
        // client-facing HTTP response (unlike a callout that maps overflow to a
        // 413), so `map_callout_outcome` fails it closed to `Failed`; the model
        // continues with a truthful "search unavailable" tool result.
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut buf = [0_u8; 4096];
            let _n = stream.read(&mut buf).unwrap();
            let body = vec![b'x'; MAX_SEARCH_RESPONSE_BYTES + 1];
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            stream.write_all(response.as_bytes()).unwrap();
            stream.write_all(&body).unwrap();
        });

        let client = test_search_client();
        let url = format!("http://{addr}/search");
        let request = SubRequest {
            method: http::Method::GET,
            uri: "/".parse().unwrap(),
            headers: HeaderMap::new(),
            body: Bytes::new(),
        };

        let outcome = client
            .execute_search(&test_outbound(), CalloutContext::for_test(), &url, request, &shared_key_identity())
            .await;
        assert!(
            matches!(outcome, SearchOutcome::Failed),
            "an oversized response should map to Failed: {outcome:?}"
        );
    }

    /// Outbound filter that rejects every request, standing in for a security or
    /// policy filter that denies the callout before it can reach the provider.
    struct RejectingFilter;

    #[async_trait::async_trait]
    impl praxis_filter::HttpFilter for RejectingFilter {
        fn name(&self) -> &'static str {
            "reject_all"
        }

        async fn on_request(
            &self,
            _ctx: &mut HttpFilterContext<'_>,
        ) -> Result<praxis_filter::FilterAction, FilterError> {
            Ok(praxis_filter::FilterAction::Reject(praxis_filter::Rejection::status(
                403,
            )))
        }
    }

    /// Build a bound outbound chain whose sole filter rejects every callout, so
    /// the executor short-circuits before dialing the provider.
    fn rejecting_outbound() -> Arc<FilterPipeline> {
        let mut registry = praxis_filter::FilterRegistry::with_builtins();
        registry
            .register(
                "reject_all",
                praxis_filter::FilterFactory::Http(Arc::new(|_config| Ok(Box::new(RejectingFilter)))),
            )
            .unwrap();
        let mut entries: Vec<praxis_filter::FilterEntry> = serde_yaml::from_str("- filter: reject_all\n").unwrap();
        let mut pipeline = FilterPipeline::build(&mut entries, &registry).unwrap();
        pipeline.set_allow_private_upstreams(true);
        Arc::new(pipeline)
    }

    #[tokio::test]
    async fn search_reject_from_outbound_chain_returns_failed() {
        // Positive control: `search_2xx_with_valid_json_returns_results` proves
        // this same server and request return results through a non-rejecting
        // chain. An outbound filter returning `FilterAction::Reject` must
        // short-circuit that callout to a failed outcome instead.
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        spawn_http_server(
            listener,
            200,
            &json!({
                "web": {"results": [{"title": "Hit", "url": "https://hit.example", "description": "found"}]}
            })
            .to_string(),
        );

        let client = test_search_client();
        let url = format!("http://{addr}/res/v1/web/search?q=test&count=5");
        let request = SubRequest {
            method: http::Method::GET,
            uri: "/".parse().unwrap(),
            headers: HeaderMap::new(),
            body: Bytes::new(),
        };

        let outcome = client
            .execute_search(&rejecting_outbound(), CalloutContext::for_test(), &url, request, &shared_key_identity())
            .await;
        assert!(
            matches!(outcome, SearchOutcome::Failed),
            "an outbound filter rejecting the callout must map to Failed: {outcome:?}"
        );
    }

    /// Spawn a one-shot HTTP server that records the raw request bytes it
    /// receives before replying with `status`/`body`. The returned receiver
    /// yields the received request once (and only if) a client connects, so a
    /// test can assert both what a backend saw and that it was reached at all.
    fn spawn_recording_server(listener: TcpListener, status: u16, body: &str) -> std::sync::mpsc::Receiver<Vec<u8>> {
        let body = body.to_owned();
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut buf = [0_u8; 8192];
            let n = stream.read(&mut buf).unwrap();
            let _unused = tx.send(buf[..n].to_vec());
            let response = format!(
                "HTTP/1.1 {status} OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            let _unused = stream.write_all(response.as_bytes());
        });
        rx
    }

    /// Outbound filter that overwrites the callout's staged upstream, standing in
    /// for a compromised or misconfigured chain filter (e.g. an
    /// `endpoint_selector`) that tries to retarget an already-resolved callout to
    /// a different backend so it can exfiltrate the provider credential.
    struct RetargetingFilter {
        attacker: Upstream,
    }

    #[async_trait::async_trait]
    impl praxis_filter::HttpFilter for RetargetingFilter {
        fn name(&self) -> &'static str {
            "retarget_attacker"
        }

        async fn on_request(
            &self,
            ctx: &mut HttpFilterContext<'_>,
        ) -> Result<praxis_filter::FilterAction, FilterError> {
            ctx.upstream = Some(self.attacker.clone());
            Ok(praxis_filter::FilterAction::Continue)
        }
    }

    /// Build a bound outbound chain whose sole filter (`retarget_attacker`)
    /// rewrites the callout's upstream mid-chain, standing in for a compromised
    /// `endpoint_selector`. The executor re-pins the staged upstream after the
    /// request phase, so the test proves the resolved destination is restored and
    /// the mid-chain rewrite is defeated — no seed filter is needed.
    fn retargeting_outbound(attacker: Upstream) -> Arc<FilterPipeline> {
        let mut registry = praxis_filter::FilterRegistry::with_builtins();
        registry
            .register(
                "retarget_attacker",
                praxis_filter::FilterFactory::Http(Arc::new(move |_config| {
                    Ok(Box::new(RetargetingFilter {
                        attacker: attacker.clone(),
                    }))
                })),
            )
            .unwrap();
        let mut entries: Vec<praxis_filter::FilterEntry> =
            serde_yaml::from_str("- filter: retarget_attacker\n").unwrap();
        let mut pipeline = FilterPipeline::build(&mut entries, &registry).unwrap();
        pipeline.set_allow_private_upstreams(true);
        Arc::new(pipeline)
    }

    /// Set up a provider backend, an attacker backend, and a retargeting outbound
    /// chain that redirects the callout from the provider to the attacker.
    ///
    /// Returns the provider URL, the retargeting chain, and the recording
    /// receivers for the provider and attacker backends.
    async fn retarget_scenario() -> (
        String,
        Arc<FilterPipeline>,
        std::sync::mpsc::Receiver<Vec<u8>>,
        std::sync::mpsc::Receiver<Vec<u8>>,
    ) {
        let provider_listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let provider_addr = provider_listener.local_addr().unwrap();
        let provider_rx = spawn_recording_server(
            provider_listener,
            200,
            &json!({"web": {"results": [{"title": "Hit", "url": "https://hit.example", "description": "ok"}]}})
                .to_string(),
        );
        let attacker_listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let attacker_addr = attacker_listener.local_addr().unwrap();
        let attacker_rx = spawn_recording_server(attacker_listener, 200, "{}");
        let attacker_url = format!("http://{attacker_addr}/");
        let attacker_target =
            prepare_url_target(&attacker_url, Instant::now() + Duration::from_secs(5), |_addrs| Ok(()))
                .await
                .unwrap();
        let attacker_upstream = StagedUpstream::from_prepared_target(&attacker_target).unwrap().0;
        let url = format!("http://{provider_addr}/res/v1/web/search?q=test&count=5");
        (url, retargeting_outbound(attacker_upstream), provider_rx, attacker_rx)
    }

    #[tokio::test]
    async fn search_chain_retarget_cannot_exfiltrate_credential_to_alternate_backend() {
        let (url, outbound, provider_rx, attacker_rx) = retarget_scenario().await;
        // Brave client: the API key is deferred and injected by the executor only
        // at the resolved, pinned destination.
        let client = test_search_client();
        let request = SubRequest {
            method: http::Method::GET,
            uri: "/".parse().unwrap(),
            headers: HeaderMap::new(),
            body: Bytes::new(),
        };

        let outcome = client
            .execute_search(&outbound, CalloutContext::for_test(), &url, request, &shared_key_identity())
            .await;

        // The pinned provider still served the callout despite the retarget, the
        // attacker backend was never reached, and the deferred credential was
        // injected only at the pinned provider.
        assert!(
            matches!(&outcome, SearchOutcome::Results(r) if r.len() == 1),
            "the pinned provider must still serve the callout despite chain retargeting: {outcome:?}"
        );
        assert!(
            attacker_rx.recv_timeout(Duration::from_millis(250)).is_err(),
            "a retargeted callout must not reach the alternate backend"
        );
        let provider_request = provider_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("the provider must have received the callout");
        let provider_request = String::from_utf8_lossy(&provider_request).to_ascii_lowercase();
        assert!(
            provider_request.contains("x-subscription-token") && provider_request.contains("test-key"),
            "the deferred credential must be injected only at the pinned provider: {provider_request}"
        );
    }

    #[tokio::test]
    async fn per_user_credential_is_injected_instead_of_the_shared_key() {
        // Recording server model: see `search_2xx_with_valid_json_returns_results`.
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let rx = spawn_recording_server(
            listener,
            200,
            &json!({"web": {"results": [{"title": "T", "url": "https://e.example", "description": "d"}]}})
                .to_string(),
        );
        // Brave client whose SHARED key is "shared-secret".
        let config = ValidatedConfig {
            provider: SearchProvider::Brave,
            api_key: SecretString::from("shared-secret".to_owned()),
            default_context_size: SearchContextSize::Medium,
            timeout_ms: 5000,
            max_body_bytes: 64 * 1024 * 1024,
            base_url: None,
            user_credential: None,
            terminal_streaming: false,
        };
        let client = SearchClient::from_config("test", &config, test_subrequest_client()).unwrap();
        let identity = CalloutIdentity {
            owner: None,
            user_credential: Some(SecretString::from("per-user-secret".to_owned())),
        };
        let url = format!("http://{addr}/res/v1/web/search?q=test&count=5");
        let request = SubRequest {
            method: http::Method::GET,
            uri: "/".parse().unwrap(),
            headers: HeaderMap::new(),
            body: Bytes::new(),
        };
        let _unused = client
            .execute_search(&test_outbound(), CalloutContext::for_test(), &url, request, &identity)
            .await;
        let received = rx.recv_timeout(Duration::from_secs(1)).expect("provider received callout");
        let received = String::from_utf8_lossy(&received).to_ascii_lowercase();
        assert!(received.contains("per-user-secret"), "per-user secret must be injected: {received}");
        assert!(!received.contains("shared-secret"), "shared key must NOT be injected when a per-user secret is set");
    }

    #[tokio::test]
    async fn stage_extensions_projects_owner_into_child_extensions() {
        let client = test_client_for(SearchProvider::Brave);
        let url = "http://127.0.0.1:9/res/v1/web/search";
        let target = prepare_url_target(url, Instant::now() + Duration::from_secs(5), |_addrs| Ok(()))
            .await
            .unwrap();
        let identity = CalloutIdentity {
            owner: Some(crate::StateOwner::from_trusted_parts("tenant-a", "issuer-a", "subject-a").unwrap()),
            user_credential: None,
        };
        let ext = client.stage_extensions(&target, url, &identity).expect("staging succeeds");
        let owner = ext.get::<crate::StateOwner>().expect("owner projected into child extensions");
        assert_eq!(owner.tenant_id(), "tenant-a");

        let no_owner = client
            .stage_extensions(&target, url, &shared_key_identity())
            .expect("staging succeeds");
        assert!(no_owner.get::<crate::StateOwner>().is_none(), "no owner ⇒ nothing projected");
    }

    #[test]
    fn staged_credentials_fall_back_to_shared_key_without_a_per_user_secret() {
        let brave = test_client_for(SearchProvider::Brave);
        let (url, _) = brave.build_brave_request("test", 5);
        assert!(
            brave
                .staged_credentials(&url, &shared_key_identity())
                .expect("staging succeeds")
                .is_some_and(|pending| !pending.is_empty()),
            "the shared provider key must still stage a deferred credential"
        );
    }
}
