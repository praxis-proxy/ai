// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Search provider abstraction and implementations.
//!
//! Uses [`CalloutClient`] from praxis-core for HTTP callouts with
//! circuit breaking, timeout, and loop prevention.

use praxis_core::callout::{
    CalloutClient, CalloutConfig, CalloutRequest, CalloutResult, CircuitBreakerConfig, FailureMode as CoreFailureMode,
};
use praxis_filter::FilterError;
use secrecy::{ExposeSecret as _, SecretString};
use serde_json::Value;
use tracing::{debug, warn};

use super::config::{FailureMode, SearchContextSize, SearchProvider, ValidatedConfig};

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

/// HTTP search client wrapping a [`CalloutClient`].
pub(crate) struct SearchClient {
    /// The underlying HTTP callout client.
    client: CalloutClient,
    /// Search backend provider.
    provider: SearchProvider,
    /// API key for the search provider.
    api_key: SecretString,
    /// Default search context size.
    default_context_size: SearchContextSize,
    /// Failure mode governing what happens on parse errors.
    failure_mode: FailureMode,
    /// HTTP status to return on rejection.
    status_on_error: u16,
}

impl std::fmt::Debug for SearchClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SearchClient")
            .field("client", &self.client)
            .field("provider", &self.provider)
            .field("api_key", &"[REDACTED]")
            .field("default_context_size", &self.default_context_size)
            .field("failure_mode", &self.failure_mode)
            .field("status_on_error", &self.status_on_error)
            .finish()
    }
}

/// Build a [`CalloutConfig`] from validated filter config.
fn build_callout_config(config: &ValidatedConfig) -> CalloutConfig {
    let failure_mode = match config.failure_mode {
        FailureMode::Closed => CoreFailureMode::Closed,
        FailureMode::Open => CoreFailureMode::Open,
    };
    CalloutConfig {
        circuit_breaker: Some(CircuitBreakerConfig {
            consecutive_failures: 5,
            recovery_window_ms: 30_000,
        }),
        failure_mode,
        status_on_error: config.status_on_error,
        timeout_ms: config.timeout_ms,
        ..CalloutConfig::default()
    }
}

impl SearchClient {
    /// Build a search client from validated filter config.
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] if the underlying [`CalloutClient`]
    /// cannot be constructed.
    pub(crate) fn from_config(config: &ValidatedConfig) -> Result<Self, FilterError> {
        let callout_config = build_callout_config(config);
        let client = CalloutClient::new(callout_config).map_err(|e| FilterError::from(e.to_string()))?;
        Ok(Self {
            client,
            provider: config.provider,
            api_key: config.api_key.clone(),
            default_context_size: config.default_context_size,
            failure_mode: config.failure_mode,
            status_on_error: config.status_on_error,
        })
    }

    /// Execute a web search query.
    pub(crate) async fn search(&self, query: &str, context_size: Option<SearchContextSize>) -> SearchOutcome {
        let size = context_size.unwrap_or(self.default_context_size);
        let count = size.result_count();
        debug!(provider = self.provider.as_str(), query, count, "executing web search");
        let request = match self.provider {
            SearchProvider::Brave => self.build_brave_request(query, count),
            SearchProvider::Tavily => self.build_tavily_request(query, size),
            SearchProvider::You => self.build_you_request(query, count),
        };
        self.handle_callout_result(self.client.execute(request).await)
    }

    /// Map a [`CalloutResult`] to a [`SearchOutcome`].
    fn handle_callout_result(&self, result: CalloutResult) -> SearchOutcome {
        match result {
            CalloutResult::Success(response) => self.parse_response(&response.body),
            CalloutResult::Failed => {
                warn!(provider = self.provider.as_str(), "search callout failed (open mode)");
                SearchOutcome::Skipped
            },
            CalloutResult::Rejected(rejection) => {
                warn!(
                    provider = self.provider.as_str(),
                    status = rejection.status,
                    "search callout rejected"
                );
                SearchOutcome::Rejected {
                    status: rejection.status,
                }
            },
        }
    }

    /// Build a Brave Search API request.
    fn build_brave_request(&self, query: &str, count: u32) -> CalloutRequest {
        let encoded_query = percent_encoding::utf8_percent_encode(query, percent_encoding::NON_ALPHANUMERIC);
        let url = format!("https://api.search.brave.com/res/v1/web/search?q={encoded_query}&count={count}");

        CalloutRequest {
            method: http::Method::GET,
            url,
            headers: vec![
                (http::header::ACCEPT, http::HeaderValue::from_static("application/json")),
                (
                    http::HeaderName::from_static("x-subscription-token"),
                    http::HeaderValue::from_str(self.api_key.expose_secret())
                        .unwrap_or_else(|_| http::HeaderValue::from_static("")),
                ),
            ],
            body: None,
            depth: 0,
        }
    }

    /// Build a Tavily Search API request.
    fn build_tavily_request(&self, query: &str, context_size: SearchContextSize) -> CalloutRequest {
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

        CalloutRequest {
            method: http::Method::POST,
            url: "https://api.tavily.com/search".to_owned(),
            headers: vec![
                (
                    http::header::CONTENT_TYPE,
                    http::HeaderValue::from_static("application/json"),
                ),
                (http::header::ACCEPT, http::HeaderValue::from_static("application/json")),
            ],
            body: Some(serde_json::to_vec(&body).unwrap_or_default()),
            depth: 0,
        }
    }

    /// Build a You.com Search API request.
    fn build_you_request(&self, query: &str, count: u32) -> CalloutRequest {
        let body = serde_json::json!({
            "query": query,
            "count": count,
        });

        CalloutRequest {
            method: http::Method::POST,
            url: "https://api.you.com/v1/search".to_owned(),
            headers: vec![
                (
                    http::header::CONTENT_TYPE,
                    http::HeaderValue::from_static("application/json"),
                ),
                (http::header::ACCEPT, http::HeaderValue::from_static("application/json")),
                (
                    http::HeaderName::from_static("x-api-key"),
                    http::HeaderValue::from_str(self.api_key.expose_secret())
                        .unwrap_or_else(|_| http::HeaderValue::from_static("")),
                ),
            ],
            body: Some(serde_json::to_vec(&body).unwrap_or_default()),
            depth: 0,
        }
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
    use secrecy::SecretString;
    use serde_json::json;

    use super::*;

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
    fn build_you_request_uses_callout_client_contract() {
        let config = ValidatedConfig {
            provider: SearchProvider::You,
            api_key: SecretString::from("test-key".to_owned()),
            default_context_size: SearchContextSize::Medium,
            timeout_ms: 5000,
            max_body_bytes: 64 * 1024 * 1024,
            failure_mode: FailureMode::Closed,
            status_on_error: 502,
        };
        let client = SearchClient::from_config(&config).unwrap();

        let request = client.build_you_request("Praxis proxy", 5);

        assert_eq!(request.method, http::Method::POST);
        assert_eq!(request.url, "https://api.you.com/v1/search");
        assert!(
            request
                .headers
                .iter()
                .any(|(name, value)| name == "x-api-key" && value == "test-key"),
            "You.com requests must send X-API-Key through the Praxis callout"
        );
        assert_eq!(
            serde_json::from_slice::<Value>(request.body.as_deref().unwrap()).unwrap(),
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
        };
        let client = SearchClient::from_config(&config);
        assert!(client.is_ok());
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
        };
        let client = SearchClient::from_config(&config).unwrap();
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
        };
        let client = SearchClient::from_config(&config).unwrap();
        let outcome = client.parse_response(b"not json");
        assert!(
            matches!(outcome, SearchOutcome::Skipped),
            "open mode should skip on parse failure"
        );
    }
}
