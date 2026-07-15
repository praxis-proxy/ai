// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Praxis-owned web-search orchestration for direct vLLM Responses backends.

#![allow(
    clippy::indexing_slicing,
    clippy::missing_docs_in_private_items,
    clippy::multiple_inherent_impl,
    clippy::too_many_lines,
    reason = "the first blocking demo keeps the orchestration contract in one focused module"
)]

use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use praxis_filter::{
    BodyAccess, BodyMode, FilterAction, FilterError, HttpFilter, HttpFilterContext, parse_filter_config,
};
use secrecy::{ExposeSecret as _, SecretString};
use serde::Deserialize;
use serde_json::{Value, json};
use tracing::debug;

const WEB_SEARCH_TYPES: &[&str] = &[
    "web_search",
    "web_search_preview",
    "web_search_preview_2025_03_11",
    "web_search_2025_08_26",
];
const MAX_BODY_BYTES: usize = 16 * 1024 * 1024;
const MAX_ERROR_BODY_CHARS: usize = 4_096;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct WebSearchConfig {
    vllm_base_url: String,
    #[serde(default = "default_you_base_url")]
    you_api_base_url: String,
    #[serde(default = "default_max_rounds")]
    max_tool_rounds: u32,
    #[serde(default = "default_timeout_secs")]
    timeout_secs: u64,
}

fn default_you_base_url() -> String {
    "https://api.you.com".to_owned()
}
fn default_max_rounds() -> u32 {
    5
}
fn default_timeout_secs() -> u64 {
    30
}

/// A blocking direct-vLLM demo filter. It owns the vLLM/You.com loop and
/// returns the completed Responses payload as an immediate response.
pub struct WebSearchOrchestratorFilter {
    config: WebSearchConfig,
    client: reqwest::Client,
    api_key: SecretString,
}

impl WebSearchOrchestratorFilter {
    /// Build the direct-vLLM orchestrator from YAML configuration.
    ///
    /// # Errors
    ///
    /// Returns an error when configuration parsing, validation, or HTTP client
    /// construction fails.
    pub fn from_config(config: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
        let cfg: WebSearchConfig = parse_filter_config("openai_web_search", config)?;
        if cfg.vllm_base_url.trim().is_empty() || cfg.max_tool_rounds < 2 || cfg.timeout_secs == 0 {
            return Err(
                "openai_web_search requires a non-empty vllm_base_url, max_tool_rounds >= 2, and timeout_secs > 0"
                    .into(),
            );
        }
        let api_key = std::env::var("YOU_API_KEY")
            .map(|value| value.trim().to_owned())
            .ok()
            .filter(|value| !value.is_empty())
            .map(SecretString::from)
            .ok_or_else(|| FilterError::from("openai_web_search requires YOU_API_KEY at startup"))?;
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(cfg.timeout_secs))
            .build()
            .map_err(|e| format!("openai_web_search client: {e}"))?;
        Ok(Box::new(Self {
            config: cfg,
            client,
            api_key,
        }))
    }
}

#[async_trait]
impl HttpFilter for WebSearchOrchestratorFilter {
    fn name(&self) -> &'static str {
        "openai_web_search"
    }

    fn request_body_access(&self) -> BodyAccess {
        BodyAccess::ReadWrite
    }

    fn request_body_mode(&self) -> BodyMode {
        BodyMode::StreamBuffer {
            max_bytes: Some(MAX_BODY_BYTES),
        }
    }

    async fn on_request(&self, _ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        Ok(FilterAction::Continue)
    }

    async fn on_request_body(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        body: &mut Option<Bytes>,
        end_of_stream: bool,
    ) -> Result<FilterAction, FilterError> {
        if !end_of_stream || ctx.request.uri.path() != "/v1/responses" {
            return Ok(FilterAction::Continue);
        }
        let Some(bytes) = body.as_ref() else {
            return Ok(FilterAction::Continue);
        };
        let request: Value =
            serde_json::from_slice(bytes).map_err(|e| format!("openai_web_search request JSON: {e}"))?;
        if !has_web_search(&request) {
            return Ok(FilterAction::Continue);
        }
        if request.get("stream").and_then(Value::as_bool) == Some(true) {
            let body = serde_json::to_vec(&json!({
                "error": {
                    "message": "streaming web_search orchestration is not supported by this blocking filter"
                }
            }))
            .map_err(|e| format!("openai_web_search error JSON: {e}"))?;
            return Ok(FilterAction::Reject(
                praxis_filter::Rejection::status(501)
                    .with_header("content-type", "application/json")
                    .with_body(body),
            ));
        }

        let response = self
            .orchestrate(request, self.api_key.expose_secret())
            .await
            .map_err(FilterError::from)?;
        let serialized = serde_json::to_vec(&response).map_err(|e| format!("openai_web_search response JSON: {e}"))?;
        let rejection = praxis_filter::Rejection::status(200)
            .with_header("content-type", "application/json")
            .with_body(serialized);
        Ok(FilterAction::Reject(rejection))
    }
}

impl WebSearchOrchestratorFilter {
    async fn orchestrate(&self, request: Value, api_key: &str) -> Result<Value, String> {
        tokio::time::timeout(
            Duration::from_secs(self.config.timeout_secs),
            self.orchestrate_inner(request, api_key),
        )
        .await
        .map_err(|error| format!("web_search orchestration timed out: {error}"))?
    }

    async fn orchestrate_inner(&self, mut request: Value, api_key: &str) -> Result<Value, String> {
        normalize_web_search_tools_in_place(&mut request);
        request["stream"] = Value::Bool(false);
        let mut input_items = input_items(&request);

        for round in 0..self.config.max_tool_rounds {
            request["input"] = Value::Array(input_items.clone());
            let response = self
                .client
                .post(format!(
                    "{}/v1/responses",
                    self.config.vllm_base_url.trim_end_matches('/')
                ))
                .json(&request)
                .send()
                .await
                .map_err(|e| format!("vLLM request failed: {e}"))?;
            let status = response.status();
            let body = response
                .text()
                .await
                .map_err(|e| format!("vLLM response body could not be read: {e}"))?;
            if !status.is_success() {
                return Err(format!("vLLM returned HTTP {status}: {}", bounded_error_body(&body)));
            }
            let response: Value =
                serde_json::from_str(&body).map_err(|e| format!("vLLM returned invalid JSON: {e}"))?;
            let calls = search_calls(&response)?;
            if calls.is_empty() {
                return Ok(response);
            }
            debug!(round, calls = calls.len(), "executing vLLM web_search calls in Praxis");
            if round + 1 == self.config.max_tool_rounds {
                return Err("web_search tool loop exceeded max_tool_rounds".to_owned());
            }
            if let Some(output) = response.get("output").and_then(Value::as_array) {
                input_items.extend(output.iter().cloned());
            }
            let outputs = futures::future::try_join_all(calls.iter().map(|call| async {
                let call_id = call
                    .get("call_id")
                    .and_then(Value::as_str)
                    .ok_or("web_search call missing call_id")?
                    .to_owned();
                let args = call.get("arguments").ok_or("web_search call missing arguments")?;
                let output = self.you_search(args, api_key).await?;
                Ok::<_, String>((call_id, output))
            }))
            .await?;
            input_items.extend(
                outputs.into_iter().map(
                    |(call_id, output)| json!({"type":"function_call_output", "call_id":call_id, "output":output}),
                ),
            );
        }
        unreachable!("max_tool_rounds is validated during filter construction");
    }

    async fn you_search(&self, args: &Value, api_key: &str) -> Result<String, String> {
        let query = args
            .get("query")
            .and_then(Value::as_str)
            .filter(|q| !q.trim().is_empty())
            .ok_or("web_search query is required")?;
        debug!(query, "sending direct-vLLM web_search request to You.com");
        let search_request = you_search_request(args, query);
        let response = self
            .client
            .post(format!(
                "{}/v1/search",
                self.config.you_api_base_url.trim_end_matches('/')
            ))
            .header("X-API-Key", api_key)
            .json(&search_request)
            .send()
            .await
            .map_err(|e| format!("You.com request failed: {e}"))?;
        let status = response.status();
        let body = response
            .text()
            .await
            .map_err(|e| format!("You.com response body could not be read: {e}"))?;
        if !status.is_success() {
            return Err(format!("You.com returned HTTP {status}: {}", bounded_error_body(&body)));
        }
        let value: Value = serde_json::from_str(&body).map_err(|e| format!("You.com returned invalid JSON: {e}"))?;
        serde_json::to_string(&json!({"query":query, "results":value.get("results").cloned().unwrap_or(json!({"web":[],"news":[]})), "metadata":value.get("metadata").cloned().unwrap_or(Value::Null)})).map_err(|e| format!("You.com result serialization failed: {e}"))
    }
}

fn bounded_error_body(body: &str) -> String {
    let mut bounded = body.chars().take(MAX_ERROR_BODY_CHARS).collect::<String>();
    if body.chars().nth(MAX_ERROR_BODY_CHARS).is_some() {
        bounded.push_str("...");
    }
    bounded
}

fn you_search_request(args: &Value, query: &str) -> Value {
    let mut request = json!({"query": query});
    for field in [
        "count",
        "freshness",
        "country",
        "language",
        "safesearch",
        "livecrawl",
        "livecrawl_formats",
        "crawl_timeout",
        "include_domains",
        "exclude_domains",
        "boost_domains",
    ] {
        if let Some(value) = args.get(field) {
            request[field] = value.clone();
        }
    }
    request
}

fn has_web_search(body: &Value) -> bool {
    body.get("tools")
        .and_then(Value::as_array)
        .is_some_and(|tools| tools.iter().any(is_web_search_tool))
}

fn is_web_search_tool(tool: &Value) -> bool {
    tool.get("type")
        .and_then(Value::as_str)
        .is_some_and(|kind| WEB_SEARCH_TYPES.contains(&kind))
}

fn normalize_web_search_tools_in_place(body: &mut Value) {
    let Some(tools) = body.get("tools").and_then(Value::as_array) else {
        return;
    };
    let normalized = normalize_tools(tools);
    body["tools"] = Value::Array(normalized);
}

fn normalize_tools(tools: &[Value]) -> Vec<Value> {
    let mut normalized = Vec::with_capacity(tools.len());
    let mut added = false;
    for tool in tools {
        if is_web_search_tool(tool) {
            if !added {
                normalized.push(web_search_function_tool());
                added = true;
            }
        } else {
            normalized.push(tool.clone());
        }
    }
    normalized
}

#[cfg(test)]
fn normalize_web_search_tools(body: &Value) -> Vec<Value> {
    body.get("tools")
        .and_then(Value::as_array)
        .map_or_else(Vec::new, |tools| normalize_tools(tools))
}

fn web_search_function_tool() -> Value {
    json!({"type":"function", "name":"web_search", "description":"Search the public web for current information and return structured results.", "parameters":{"type":"object","properties":{"query":{"type":"string"},"count":{"type":"integer"},"freshness":{"type":"string"},"country":{"type":"string"},"language":{"type":"string"},"include_domains":{"type":"array","items":{"type":"string"}},"exclude_domains":{"type":"array","items":{"type":"string"}}},"required":["query"]},"strict":false})
}

fn search_calls(response: &Value) -> Result<Vec<Value>, String> {
    response.get("output").and_then(Value::as_array).map_or_else(
        || Ok(Vec::new()),
        |output| {
            output
                .iter()
                .filter(|item| {
                    item.get("type").and_then(Value::as_str) == Some("function_call")
                        && item.get("name").and_then(Value::as_str) == Some("web_search")
                        && item.get("status").and_then(Value::as_str) == Some("completed")
                })
                .map(|item| {
                    let args = item
                        .get("arguments")
                        .and_then(Value::as_str)
                        .ok_or("web_search arguments missing")?;
                    let arguments: Value =
                        serde_json::from_str(args).map_err(|e| format!("web_search arguments invalid JSON: {e}"))?;
                    let mut call = item.clone();
                    call["arguments"] = arguments;
                    Ok(call)
                })
                .collect()
        },
    )
}

fn input_items(body: &Value) -> Vec<Value> {
    match body.get("input") {
        Some(Value::Array(items)) => items.clone(),
        Some(Value::String(text)) => vec![json!({"type":"message", "role":"user", "content":text})],
        _ => Vec::new(),
    }
}

#[cfg(test)]
fn append_function_call_outputs(body: &mut Value, outputs: &[Value]) {
    let items = input_items(body);
    body["input_items"] = Value::Array(outputs.to_vec());
    body["input"] = Value::Array(items.into_iter().chain(outputs.iter().cloned()).collect());
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "test assertions use expect for readable failure context"
)]
mod tests {
    use bytes::Bytes;
    use http::Method;
    use praxis_filter::HttpFilter as _;
    use serde_json::{Value, json};
    use wiremock::{
        Match, Mock, MockServer, Request, ResponseTemplate,
        matchers::{method, path},
    };

    #[derive(Debug)]
    struct InputContainsReasoning;

    impl Match for InputContainsReasoning {
        fn matches(&self, request: &Request) -> bool {
            serde_json::from_slice::<Value>(&request.body)
                .ok()
                .and_then(|body| body.get("input").cloned())
                .and_then(|input| input.as_array().cloned())
                .is_some_and(|items| {
                    items
                        .iter()
                        .any(|item| item.get("type").and_then(Value::as_str) == Some("reasoning"))
                })
        }
    }

    use super::{
        WebSearchConfig, WebSearchOrchestratorFilter, append_function_call_outputs, normalize_web_search_tools,
        search_calls,
    };

    #[test]
    fn normalizes_all_seb_supported_web_search_variants() {
        let body = json!({
            "tools": [
                {"type": "web_search"},
                {"type": "web_search_preview"},
                {"type": "web_search_preview_2025_03_11"},
                {"type": "web_search_2025_08_26"}
            ]
        });

        let normalized = normalize_web_search_tools(&body);
        assert_eq!(normalized[0]["type"], "function");
        assert_eq!(normalized[0]["name"], "web_search");
        assert_eq!(normalized.len(), 1);
    }

    #[test]
    fn extracts_completed_web_search_calls() {
        let response = json!({
            "output": [
                {"type": "reasoning"},
                {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "web_search", "arguments": "{\"query\":\"latest Rust release\"}", "status": "completed"}
            ]
        });

        let calls = search_calls(&response).expect("valid call");
        assert_eq!(calls[0]["call_id"], "call_1");
        assert_eq!(calls[0]["arguments"]["query"], "latest Rust release");
    }

    #[test]
    fn bounds_provider_error_bodies() {
        let body = "x".repeat(5_000);
        let bounded = super::bounded_error_body(&body);
        assert_eq!(bounded.chars().count(), 4_099);
        assert!(bounded.ends_with("..."));
    }

    #[test]
    fn appends_function_call_outputs_without_replacing_original_input() {
        let mut body = json!({"input": "Search this", "tools": []});
        append_function_call_outputs(
            &mut body,
            &[json!({"type":"function_call_output", "call_id":"call_1", "output":"{\"query\":\"Search this\"}"})],
        );

        assert_eq!(body["input"][0]["type"], "message");
        assert_eq!(body["input"][1]["type"], "function_call_output");
        assert_eq!(body["input"][1]["call_id"], "call_1");
    }

    #[tokio::test]
    async fn orchestrates_vllm_tool_call_through_you_and_back_to_vllm() {
        let vllm = MockServer::start().await;
        let you = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/v1/search"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "results": {"web": [{"url":"https://www.rust-lang.org","title":"Rust"}], "news": []},
                "metadata": {"search_uuid":"search_1"}
            })))
            .expect(1)
            .mount(&you)
            .await;

        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "output": [{"type":"function_call","id":"fc_1","call_id":"call_1","name":"web_search","arguments":"{\"query\":\"latest Rust release\"}","status":"completed"}]
            })))
            .up_to_n_times(1)
            .mount(&vllm)
            .await;
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id":"resp_final","object":"response","status":"completed",
                "output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Rust is current."}]}]
            })))
            .mount(&vllm)
            .await;

        let filter = WebSearchOrchestratorFilter {
            config: WebSearchConfig {
                vllm_base_url: vllm.uri(),
                you_api_base_url: you.uri(),
                max_tool_rounds: 3,
                timeout_secs: 5,
            },
            client: reqwest::Client::new(),
            api_key: secrecy::SecretString::from("test-you-key"),
        };
        let response = filter
            .orchestrate(
                json!({"model":"openai/gpt-oss-20b","input":"Find the latest Rust release.","tools":[{"type":"web_search_preview"}]}),
                "test-you-key",
            )
            .await
            .expect("orchestration should complete");

        assert_eq!(response["id"], "resp_final");
    }

    #[tokio::test]
    async fn bypasses_requests_without_web_search_tools() {
        let request = crate::test_utils::make_request(Method::POST, "/v1/responses");
        let mut ctx = crate::test_utils::make_filter_context(&request);
        let filter = WebSearchOrchestratorFilter {
            config: WebSearchConfig {
                vllm_base_url: "http://unused".to_owned(),
                you_api_base_url: "http://unused".to_owned(),
                max_tool_rounds: 1,
                timeout_secs: 5,
            },
            client: reqwest::Client::new(),
            api_key: secrecy::SecretString::from("test-you-key"),
        };
        let mut body = Some(Bytes::from(
            json!({"tools": [{"type": "function", "name": "lookup"}]}).to_string(),
        ));
        let action = filter
            .on_request_body(&mut ctx, &mut body, true)
            .await
            .expect("filter should continue");
        assert!(matches!(action, praxis_filter::FilterAction::Continue));

        let request = crate::test_utils::make_request(Method::POST, "/v1/responses");
        let mut ctx = crate::test_utils::make_filter_context(&request);
        let mut body = Some(Bytes::from(
            json!({"stream": true, "tools": [{"type": "web_search_preview"}]}).to_string(),
        ));
        let action = filter
            .on_request_body(&mut ctx, &mut body, true)
            .await
            .expect("filter should reject streaming input");
        assert!(matches!(
            action,
            praxis_filter::FilterAction::Reject(rejection) if rejection.status == 501
        ));
    }

    #[tokio::test]
    async fn preserves_all_vllm_output_items_between_rounds() {
        let vllm = MockServer::start().await;
        let you = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/v1/search"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({"results": {"web": []}})))
            .mount(&you)
            .await;

        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "output": [
                    {"type":"reasoning","summary":[]},
                    {"type":"function_call","call_id":"call_1","name":"web_search","arguments":"{\"query\":\"Rust\"}","status":"completed"}
                ]
            })))
            .up_to_n_times(1)
            .mount(&vllm)
            .await;
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .and(InputContainsReasoning)
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({"id":"final","output":[]})))
            .expect(1)
            .mount(&vllm)
            .await;

        let filter = WebSearchOrchestratorFilter {
            config: WebSearchConfig {
                vllm_base_url: vllm.uri(),
                you_api_base_url: you.uri(),
                max_tool_rounds: 3,
                timeout_secs: 5,
            },
            client: reqwest::Client::new(),
            api_key: secrecy::SecretString::from("test-you-key"),
        };
        let response = filter
            .orchestrate(
                json!({"input":"Search","tools":[{"type":"web_search_preview"}]}),
                "test-you-key",
            )
            .await
            .expect("orchestration should complete");
        assert_eq!(response["id"], "final");
    }

    #[tokio::test]
    async fn returns_vllm_error_details() {
        let vllm = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(ResponseTemplate::new(422).set_body_string("invalid model"))
            .mount(&vllm)
            .await;
        let filter = WebSearchOrchestratorFilter {
            config: WebSearchConfig {
                vllm_base_url: vllm.uri(),
                you_api_base_url: "http://unused".to_owned(),
                max_tool_rounds: 1,
                timeout_secs: 5,
            },
            client: reqwest::Client::new(),
            api_key: secrecy::SecretString::from("test-you-key"),
        };
        let error = filter
            .orchestrate(json!({"tools":[{"type":"web_search_preview"}]}), "test-you-key")
            .await
            .expect_err("vLLM error should propagate");
        assert!(error.contains("422") && error.contains("invalid model"));
    }

    #[tokio::test]
    async fn returns_you_error_details() {
        let vllm = MockServer::start().await;
        let you = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "output": [{"type":"function_call","call_id":"call_1","name":"web_search","arguments":"{\"query\":\"Rust\"}","status":"completed"}]
            })))
            .mount(&vllm)
            .await;
        Mock::given(method("POST"))
            .and(path("/v1/search"))
            .respond_with(ResponseTemplate::new(429).set_body_string("rate limited"))
            .mount(&you)
            .await;
        let filter = WebSearchOrchestratorFilter {
            config: WebSearchConfig {
                vllm_base_url: vllm.uri(),
                you_api_base_url: you.uri(),
                max_tool_rounds: 2,
                timeout_secs: 5,
            },
            client: reqwest::Client::new(),
            api_key: secrecy::SecretString::from("test-you-key"),
        };
        let error = filter
            .orchestrate(json!({"tools":[{"type":"web_search_preview"}]}), "test-you-key")
            .await
            .expect_err("You.com error should propagate");
        assert!(error.contains("429") && error.contains("rate limited"));
    }

    #[tokio::test]
    async fn enforces_max_tool_rounds() {
        let vllm = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "output": [{"type":"function_call","call_id":"call_1","name":"web_search","arguments":"{\"query\":\"Rust\"}","status":"completed"}]
            })))
            .mount(&vllm)
            .await;
        let filter = WebSearchOrchestratorFilter {
            config: WebSearchConfig {
                vllm_base_url: vllm.uri(),
                you_api_base_url: "http://unused".to_owned(),
                max_tool_rounds: 1,
                timeout_secs: 5,
            },
            client: reqwest::Client::new(),
            api_key: secrecy::SecretString::from("test-you-key"),
        };
        let error = filter
            .orchestrate(json!({"tools":[{"type":"web_search_preview"}]}), "test-you-key")
            .await
            .expect_err("tool loop should stop at max_tool_rounds");
        assert!(error.contains("max_tool_rounds"));
    }
}
