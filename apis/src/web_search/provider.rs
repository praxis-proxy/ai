// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Search provider abstraction and implementations.
//!
//! Uses [`SubRequestClient`] from praxis-core for HTTP callouts
//! with connection pooling, admission control, and TLS.
//!
//! [`SubRequestClient`]: praxis_core::subrequest::SubRequestClient

use std::time::Duration;

use bytes::Bytes;
use http::HeaderMap;
use praxis_filter::FilterError;
use secrecy::{ExposeSecret as _, SecretString};
use serde_json::Value;
use tracing::{debug, warn};

use super::{
    ValidatedConfig,
    config::{FailureMode, SearchContextSize, SearchProvider},
};
use crate::subrequest::{self, SubRequest, SubRequestClient, SubRequestError, SubResponse};

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
    /// Search succeeded with results.
    Results(Vec<SearchResult>),
    /// Search failed but failure mode is open — continue without results.
    Skipped,
    /// Search failed and failure mode is closed — reject the request.
    Rejected {
        /// HTTP status code to return.
        status: u16,
    },
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
    /// Failure mode governing what happens on errors.
    failure_mode: FailureMode,
    /// HTTP status to return on rejection.
    status_on_error: u16,
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
            .field("failure_mode", &self.failure_mode)
            .field("status_on_error", &self.status_on_error)
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
            failure_mode: config.failure_mode,
            status_on_error: config.status_on_error,
            base_url: config.base_url.clone(),
        })
    }

    /// Execute a web search query.
    pub(crate) async fn search(&self, query: &str, context_size: Option<SearchContextSize>) -> SearchOutcome {
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
        self.execute_search(&url, request).await
    }

    /// Execute a search request and map the result to a
    /// [`SearchOutcome`].
    async fn execute_search(&self, url: &str, request: SubRequest) -> SearchOutcome {
        let result = subrequest::execute_url(&self.client, url, request, MAX_SEARCH_RESPONSE_BYTES, self.timeout).await;
        self.map_search_result(result)
    }

    /// Map a sub-request result to a [`SearchOutcome`].
    fn map_search_result(&self, result: Result<SubResponse, SubRequestError>) -> SearchOutcome {
        match result {
            Ok(response) if (200..300).contains(&(response.status as usize)) => self.parse_response(&response.body),
            Ok(response) => {
                warn!(
                    provider = self.provider.as_str(),
                    status = response.status,
                    "search callout returned non-2xx"
                );
                self.transport_failure_outcome()
            },
            Err(e) => {
                warn!(provider = self.provider.as_str(), error = %e, "search callout failed");
                self.transport_failure_outcome()
            },
        }
    }

    /// Outcome for a transport or non-2xx failure. Under closed
    /// mode this is a rejection; under open mode search is silently
    /// skipped.
    fn transport_failure_outcome(&self) -> SearchOutcome {
        match self.failure_mode {
            FailureMode::Closed => SearchOutcome::Rejected {
                status: self.status_on_error,
            },
            FailureMode::Open => SearchOutcome::Skipped,
        }
    }

    /// Build a Brave Search API request.
    fn build_brave_request(&self, query: &str, count: u32) -> (String, SubRequest) {
        let encoded_query = percent_encoding::utf8_percent_encode(query, percent_encoding::NON_ALPHANUMERIC);
        let base = self.base_url.as_deref().unwrap_or("https://api.search.brave.com");
        let url = format!("{base}/res/v1/web/search?q={encoded_query}&count={count}");

        let mut headers = HeaderMap::new();
        headers.insert(http::header::ACCEPT, http::HeaderValue::from_static("application/json"));
        headers.insert(
            http::HeaderName::from_static("x-subscription-token"),
            http::HeaderValue::from_str(self.api_key.expose_secret())
                .unwrap_or_else(|_| http::HeaderValue::from_static("")),
        );

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

        let mut headers = HeaderMap::new();
        headers.insert(
            http::header::CONTENT_TYPE,
            http::HeaderValue::from_static("application/json"),
        );
        headers.insert(http::header::ACCEPT, http::HeaderValue::from_static("application/json"));
        headers.insert(
            http::HeaderName::from_static("x-api-key"),
            http::HeaderValue::from_str(self.api_key.expose_secret())
                .unwrap_or_else(|_| http::HeaderValue::from_static("")),
        );

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
                return self.parse_failure_outcome();
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

    /// Outcome for a response that arrived as 2xx but could not be
    /// parsed. Under closed mode this is an error; under open mode
    /// search is silently skipped.
    fn parse_failure_outcome(&self) -> SearchOutcome {
        match self.failure_mode {
            FailureMode::Closed => SearchOutcome::Rejected {
                status: self.status_on_error,
            },
            FailureMode::Open => SearchOutcome::Skipped,
        }
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
        assert!(parse_brave_results(&json).is_empty());
    }

    #[test]
    fn parse_brave_results_missing_web() {
        let json = json!({"query": "test"});
        assert!(parse_brave_results(&json).is_empty());
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
        assert!(parse_tavily_results(&json).is_empty());
    }

    #[test]
    fn parse_tavily_results_missing_results() {
        let json = json!({"answer": "some answer"});
        assert!(parse_tavily_results(&json).is_empty());
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
    fn build_you_request_sends_api_key_and_body() {
        let config = ValidatedConfig {
            provider: SearchProvider::You,
            api_key: SecretString::from("test-key".to_owned()),
            default_context_size: SearchContextSize::Medium,
            timeout_ms: 5000,
            max_body_bytes: 64 * 1024 * 1024,
            failure_mode: FailureMode::Closed,
            status_on_error: 502,
            base_url: None,
        };
        let client = SearchClient::from_config("test", &config, test_subrequest_client()).unwrap();

        let (url, request) = client.build_you_request("Praxis proxy", 5);

        assert_eq!(url, "https://api.you.com/v1/search");
        assert_eq!(request.method, http::Method::POST);
        assert!(
            request.headers.get("x-api-key").is_some_and(|v| v == "test-key"),
            "You.com requests must send X-API-Key"
        );
        assert_eq!(
            serde_json::from_slice::<Value>(&request.body).unwrap(),
            json!({"query": "Praxis proxy", "count": 5})
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
        assert!(parse_you_results(&json!({"results": {}})).is_empty());
    }

    #[test]
    fn search_client_from_config() {
        let config = ValidatedConfig {
            provider: SearchProvider::Brave,
            api_key: SecretString::from("test-key".to_owned()),
            default_context_size: SearchContextSize::Medium,
            timeout_ms: 5000,
            max_body_bytes: 64 * 1024 * 1024,
            failure_mode: FailureMode::Closed,
            status_on_error: 502,
            base_url: None,
        };
        let client = SearchClient::from_config("test", &config, test_subrequest_client());
        assert!(client.is_ok());
    }

    #[test]
    fn invalid_api_key_diagnostic_names_owner() {
        let config = ValidatedConfig {
            provider: SearchProvider::Brave,
            api_key: SecretString::from("invalid\nkey".to_owned()),
            default_context_size: SearchContextSize::Medium,
            timeout_ms: 5000,
            max_body_bytes: 64 * 1024 * 1024,
            failure_mode: FailureMode::Closed,
            status_on_error: 502,
            base_url: None,
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
            failure_mode: FailureMode::Closed,
            status_on_error: 502,
            base_url: Some("http://localhost:9999".into()),
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
            failure_mode: FailureMode::Closed,
            status_on_error: 502,
            base_url: Some("http://localhost:9999".into()),
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
            failure_mode: FailureMode::Closed,
            status_on_error: 502,
            base_url: Some("http://localhost:9999".into()),
        };
        let client = SearchClient::from_config("test", &config, test_subrequest_client()).unwrap();
        let (url, _) = client.build_you_request("test query", 5);
        assert!(
            url.starts_with("http://localhost:9999/"),
            "base_url should override the default You.com URL; got: {url}"
        );
    }

    #[test]
    fn parse_failure_closed_mode_rejects() {
        let config = ValidatedConfig {
            provider: SearchProvider::Brave,
            api_key: SecretString::from("test-key".to_owned()),
            default_context_size: SearchContextSize::Medium,
            timeout_ms: 5000,
            max_body_bytes: 64 * 1024 * 1024,
            failure_mode: FailureMode::Closed,
            status_on_error: 502,
            base_url: None,
        };
        let client = SearchClient::from_config("test", &config, test_subrequest_client()).unwrap();
        let outcome = client.parse_response(b"not json");
        assert!(
            matches!(outcome, SearchOutcome::Rejected { status: 502 }),
            "closed mode should reject on parse failure"
        );
    }

    #[test]
    fn parse_failure_open_mode_skips() {
        let config = ValidatedConfig {
            provider: SearchProvider::Brave,
            api_key: SecretString::from("test-key".to_owned()),
            default_context_size: SearchContextSize::Medium,
            timeout_ms: 5000,
            max_body_bytes: 64 * 1024 * 1024,
            failure_mode: FailureMode::Open,
            status_on_error: 502,
            base_url: None,
        };
        let client = SearchClient::from_config("test", &config, test_subrequest_client()).unwrap();
        let outcome = client.parse_response(b"not json");
        assert!(
            matches!(outcome, SearchOutcome::Skipped),
            "open mode should skip on parse failure"
        );
    }

    fn test_search_client(failure_mode: FailureMode) -> SearchClient {
        let config = ValidatedConfig {
            provider: SearchProvider::Brave,
            api_key: SecretString::from("test-key".to_owned()),
            default_context_size: SearchContextSize::Medium,
            timeout_ms: 1000,
            max_body_bytes: 64 * 1024 * 1024,
            failure_mode,
            status_on_error: 502,
            base_url: None,
        };
        SearchClient::from_config("test", &config, test_subrequest_client()).unwrap()
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

        let client = test_search_client(FailureMode::Closed);
        let url = format!("http://{addr}/res/v1/web/search?q=test&count=5");
        let request = SubRequest {
            method: http::Method::GET,
            uri: "/".parse().unwrap(),
            headers: HeaderMap::new(),
            body: Bytes::new(),
        };

        let outcome = client.execute_search(&url, request).await;
        assert!(
            matches!(&outcome, SearchOutcome::Results(r) if r.len() == 1),
            "2xx with valid JSON should return results: {outcome:?}"
        );
    }

    #[tokio::test]
    async fn search_non_2xx_closed_rejects() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        spawn_http_server(listener, 500, "internal error");

        let client = test_search_client(FailureMode::Closed);
        let url = format!("http://{addr}/search");
        let request = SubRequest {
            method: http::Method::GET,
            uri: "/".parse().unwrap(),
            headers: HeaderMap::new(),
            body: Bytes::new(),
        };

        let outcome = client.execute_search(&url, request).await;
        assert!(
            matches!(outcome, SearchOutcome::Rejected { status: 502 }),
            "non-2xx under closed mode should reject: {outcome:?}"
        );
    }

    #[tokio::test]
    async fn search_non_2xx_open_skips() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        spawn_http_server(listener, 429, "rate limited");

        let client = test_search_client(FailureMode::Open);
        let url = format!("http://{addr}/search");
        let request = SubRequest {
            method: http::Method::GET,
            uri: "/".parse().unwrap(),
            headers: HeaderMap::new(),
            body: Bytes::new(),
        };

        let outcome = client.execute_search(&url, request).await;
        assert!(
            matches!(outcome, SearchOutcome::Skipped),
            "non-2xx under open mode should skip: {outcome:?}"
        );
    }

    #[tokio::test]
    async fn search_connection_failure_closed_rejects() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            drop(stream);
        });

        let client = test_search_client(FailureMode::Closed);
        let url = format!("http://{addr}/search");
        let request = SubRequest {
            method: http::Method::GET,
            uri: "/".parse().unwrap(),
            headers: HeaderMap::new(),
            body: Bytes::new(),
        };

        let outcome = client.execute_search(&url, request).await;
        assert!(
            matches!(outcome, SearchOutcome::Rejected { status: 502 }),
            "connection failure under closed mode should reject: {outcome:?}"
        );
    }

    #[tokio::test]
    async fn search_timeout_open_skips() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (_stream, _) = listener.accept().await.unwrap();
            tokio::time::sleep(Duration::from_secs(5)).await;
        });

        let mut client = test_search_client(FailureMode::Open);
        client.timeout = Duration::from_millis(50);
        let url = format!("http://{addr}/search");
        let request = SubRequest {
            method: http::Method::GET,
            uri: "/".parse().unwrap(),
            headers: HeaderMap::new(),
            body: Bytes::new(),
        };

        let outcome = client.execute_search(&url, request).await;
        assert!(
            matches!(outcome, SearchOutcome::Skipped),
            "timeout under open mode should skip: {outcome:?}"
        );
    }

    #[tokio::test]
    async fn search_oversized_response_closed_rejects() {
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

        let client = test_search_client(FailureMode::Closed);
        let url = format!("http://{addr}/search");
        let request = SubRequest {
            method: http::Method::GET,
            uri: "/".parse().unwrap(),
            headers: HeaderMap::new(),
            body: Bytes::new(),
        };

        let outcome = client.execute_search(&url, request).await;
        assert!(
            matches!(outcome, SearchOutcome::Rejected { status: 502 }),
            "oversized response under closed mode should reject: {outcome:?}"
        );
    }
}
