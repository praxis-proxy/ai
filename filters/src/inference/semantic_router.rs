// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Semantic Router filter: evaluates prompt complexity to route requests dynamically.

use async_trait::async_trait;
use bytes::Bytes;
use praxis_filter::{BodyAccess, BodyMode, FilterAction, FilterError, HttpFilter, HttpFilterContext, parse_filter_config};
use serde::Deserialize;

// -----------------------------------------------------------------------------
// Config
// -----------------------------------------------------------------------------

/// Deserialized YAML config for the semantic router filter.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SemanticRouterConfig {
    /// Max bytes to buffer for the JSON payload.
    #[serde(default = "default_max_body_bytes")]
    max_body_bytes: usize,

    /// Evaluation backend configuration.
    #[serde(default)]
    backend: EvaluationBackend,

    /// Routing rules mapping complexity scores to targets.
    #[serde(default)]
    routes: Vec<RouteRule>,
}

#[derive(Debug, Deserialize, Clone)]
struct RouteRule {
    /// Minimum complexity score (inclusive).
    #[serde(default)]
    min_score: f32,
    /// Maximum complexity score (inclusive).
    #[serde(default = "default_max_score")]
    max_score: f32,
    /// Target Praxis cluster to route to.
    target_cluster: String,
}

fn default_max_score() -> f32 {
    1.0
}

#[derive(Debug, Deserialize, Default, Clone)]
#[serde(rename_all = "snake_case")]
enum EvaluationBackend {
    #[default]
    Mock, // For testing, always returns a low score
    Ort {
        _model_path: String,
        // tokenizer_path: String,
    },
    Http {
        url: String,
        auth_header: Option<String>,
        model_name: Option<String>, // e.g. gemini-3.1-flash-lite
    }
}

/// Default max body bytes (1MB).
fn default_max_body_bytes() -> usize {
    1024 * 1024
}

// -----------------------------------------------------------------------------
// SemanticRouterFilter
// -----------------------------------------------------------------------------

/// Intercepts AI traffic, evaluates prompt complexity, and routes dynamically.
///
/// # YAML configuration
///
/// ```yaml
/// filter: semantic_router
/// backend:
///   http:
///     url: "http://localhost:8080/v1/chat/completions"
/// routes:
///   - min_score: 0.8
///     target_cluster: heavy_engineering
///   - max_score: 0.79
///     target_cluster: fast_chat
/// max_body_bytes: 2097152 # optional, default 1MB
/// ```
///
/// # Example
///
/// ```ignore
/// use praxis_ai_filters::SemanticRouterFilter;
///
/// let yaml = serde_yaml::Value::Null;
/// let filter = SemanticRouterFilter::from_config(&yaml).unwrap();
/// assert_eq!(filter.name(), "semantic_router");
/// ```
pub struct SemanticRouterFilter {
    max_body_bytes: usize,
    backend: EvaluationBackend,
    routes: Vec<RouteRule>,
    http_client: Option<reqwest::Client>,
}

impl SemanticRouterFilter {
    /// Create from parsed YAML config.
    pub fn from_config(config: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
        let cfg: SemanticRouterConfig = parse_filter_config("semantic_router", config)?;

        let http_client = if matches!(cfg.backend, EvaluationBackend::Http { .. }) {
            Some(reqwest::Client::builder().build().map_err(|e| FilterError::from(e.to_string()))?)
        } else {
            None
        };

        Ok(Box::new(Self {
            max_body_bytes: cfg.max_body_bytes,
            backend: cfg.backend,
            routes: cfg.routes,
            http_client,
        }))
    }
}

#[async_trait]
impl HttpFilter for SemanticRouterFilter {
    fn name(&self) -> &'static str {
        "semantic_router"
    }

    async fn on_request(&self, _ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        // Implementation for interception and evaluation will go here (Phase 2 & 3).
        Ok(FilterAction::Continue)
    }

    async fn on_response(&self, _ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        Ok(FilterAction::Continue)
    }

    fn request_body_access(&self) -> BodyAccess {
        BodyAccess::ReadOnly
    }

    fn request_body_mode(&self) -> BodyMode {
        BodyMode::StreamBuffer {
            max_bytes: Some(self.max_body_bytes),
        }
    }

    async fn on_request_body(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        body: &mut Option<Bytes>,
        end_of_stream: bool,
    ) -> Result<FilterAction, FilterError> {
        if !end_of_stream {
            return Ok(FilterAction::Continue);
        }

        let Some(raw) = body.as_ref() else {
            return Ok(FilterAction::Continue);
        };

        // Parse JSON payload
        let value: serde_json::Value = match serde_json::from_slice(raw) {
            Ok(v) => v,
            Err(_) => return Ok(FilterAction::Continue),
        };

        // Extract the user's prompt (naive extraction: last message's content)
        if let Some(messages) = value.get("messages").and_then(serde_json::Value::as_array) {
            if let Some(last_message) = messages.last() {
                if let Some(content) = last_message.get("content").and_then(serde_json::Value::as_str) {
                    tracing::debug!(target: "semantic_router", "extracted prompt: {}", content);
                    
                    // Evaluate prompt complexity
                    let score = match &self.backend {
                        EvaluationBackend::Mock => 0.1, // Mock always returns low score
                        EvaluationBackend::Ort { .. } => {
                            // Local ORT inference goes here (returns mock score for now)
                            0.1
                        },
                        EvaluationBackend::Http { url, auth_header, model_name } => {
                            if let Some(client) = &self.http_client {
                                let payload = serde_json::json!({
                                    "model": model_name.as_deref().unwrap_or("gemini-3.1-flash-lite"),
                                    "messages": [{"role": "user", "content": format!("Score the complexity of this prompt from 0.0 to 1.0 (return only a float): {}", content)}]
                                });

                                let mut req = client.post(url).json(&payload);
                                if let Some(auth) = auth_header {
                                    req = req.header("Authorization", auth);
                                }

                                match req.send().await {
                                    Ok(resp) if resp.status().is_success() => {
                                        // Attempt to parse response as float (naive extraction for example)
                                        if let Ok(text) = resp.text().await {
                                            tracing::debug!(target: "semantic_router", "http eval response: {}", text);
                                            text.trim().parse::<f32>().unwrap_or(0.1)
                                        } else {
                                            0.1
                                        }
                                    }
                                    Err(e) => {
                                        tracing::error!(target: "semantic_router", "http eval error: {}", e);
                                        0.1
                                    }
                                    _ => 0.1,
                                }
                            } else {
                                0.1
                            }
                        }
                    };

                    tracing::debug!(target: "semantic_router", "prompt evaluation score: {}", score);
                    
                    // Evaluate routing rules based on score
                    for route in &self.routes {
                        if score >= route.min_score && score <= route.max_score {
                            tracing::debug!(target: "semantic_router", "routing to cluster: {}", route.target_cluster);
                            ctx.cluster = Some(std::sync::Arc::from(route.target_cluster.clone()));
                            break;
                        }
                    }
                }
            }
        }

        Ok(FilterAction::Continue)
    }
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
    use super::*;

    #[test]
    fn from_config_default_routes() {
        let filter = SemanticRouterFilter::from_config(&serde_yaml::Value::Null).unwrap();
        assert_eq!(
            filter.name(),
            "semantic_router",
            "default config should produce semantic_router"
        );
    }

    #[test]
    fn from_config_custom_routes() {
        let yaml: serde_yaml::Value = serde_yaml::from_str(r#"
routes:
  - min_score: 0.8
    target_cluster: heavy
  - max_score: 0.79
    target_cluster: light
"#).unwrap();
        let filter = SemanticRouterFilter::from_config(&yaml).unwrap();
        assert_eq!(filter.name(), "semantic_router", "custom routes config should parse");
    }

    #[test]
    fn body_access_and_mode() {
        let filter = SemanticRouterFilter::from_config(&serde_yaml::Value::Null).unwrap();
        assert_eq!(filter.request_body_access(), BodyAccess::ReadOnly);
        assert!(matches!(
            filter.request_body_mode(),
            BodyMode::StreamBuffer { max_bytes: Some(1048576) }
        ));
    }

    #[tokio::test]
    async fn on_request_body_ignores_incomplete_stream() {
        let filter = SemanticRouterFilter::from_config(&serde_yaml::Value::Null).unwrap();
        let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat/completions");
        let mut ctx = crate::test_utils::make_filter_context(&req);
        let mut body = Some(Bytes::from_static(b"{\"msg"));
        
        let action = filter.on_request_body(&mut ctx, &mut body, false).await.unwrap();
        assert!(matches!(action, FilterAction::Continue));
    }

    #[tokio::test]
    async fn on_request_body_extracts_prompt_and_routes() {
        let yaml: serde_yaml::Value = serde_yaml::from_str(r#"
routes:
  - min_score: 0.0
    max_score: 1.0
    target_cluster: test_cluster
"#).unwrap();
        let filter = SemanticRouterFilter::from_config(&yaml).unwrap();
        let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat/completions");
        let mut ctx = crate::test_utils::make_filter_context(&req);
        
        let json = br#"{"messages":[{"role":"user","content":"route me"}]}"#;
        let mut body = Some(Bytes::from_static(json));
        
        let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
        assert!(matches!(action, FilterAction::Continue));
        assert_eq!(ctx.cluster.as_deref(), Some("test_cluster"));
    }
}
