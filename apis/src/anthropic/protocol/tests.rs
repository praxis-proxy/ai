// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Unit tests for the `anthropic_messages_protocol` filter.

use super::*;

// -----------------------------------------------------------------------------
// Config Parsing
// -----------------------------------------------------------------------------

#[test]
fn default_config_parses() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("{}").unwrap();
    let filter = AnthropicMessagesProtocolFilter::from_config(&yaml).unwrap();
    assert_eq!(
        filter.name(),
        "anthropic_messages_protocol",
        "filter name should be anthropic_messages_protocol"
    );
}

#[test]
fn empty_version_rejected() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("default_version: \"\"").unwrap();
    let result = AnthropicMessagesProtocolFilter::from_config(&yaml);
    assert!(result.is_err(), "empty default_version should be rejected");
}

#[test]
fn invalid_header_value_rejected() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("default_version: \"2023-06-01\\nmalformed\"").unwrap();
    let result = AnthropicMessagesProtocolFilter::from_config(&yaml);
    assert!(
        result.is_err(),
        "invalid default_version header value should be rejected"
    );
}

// -----------------------------------------------------------------------------
// Header Injection
// -----------------------------------------------------------------------------

#[tokio::test]
async fn injects_anthropic_version_when_absent() {
    let filter = make_filter("{}");
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/messages");

    let req: &'static praxis_filter::Request = Box::leak(Box::new(req));
    let mut ctx = crate::test_utils::make_filter_context(req);

    let action = filter.on_request(&mut ctx).await.unwrap();
    assert!(matches!(action, FilterAction::Continue), "filter should continue");

    let headers: std::collections::HashMap<&str, &str> = ctx
        .extra_request_headers
        .iter()
        .map(|(k, v)| (k.as_ref(), v.as_str()))
        .collect();

    assert_eq!(
        headers.get("anthropic-version"),
        Some(&"2023-06-01"),
        "should inject default anthropic-version"
    );
}

#[tokio::test]
async fn does_not_inject_when_header_present() {
    let filter = make_filter("{}");
    let mut req = crate::test_utils::make_request(http::Method::POST, "/v1/messages");
    req.headers.insert("anthropic-version", "2024-01-01".parse().unwrap());

    let req: &'static praxis_filter::Request = Box::leak(Box::new(req));
    let mut ctx = crate::test_utils::make_filter_context(req);

    let _action = filter.on_request(&mut ctx).await.unwrap();

    assert!(
        ctx.extra_request_headers.is_empty(),
        "should not inject when header already present"
    );
}

/// A translation-owned queued version must suppress native defaults.
#[tokio::test]
async fn does_not_inject_when_version_set_is_pending() {
    let filter = make_filter("{}");
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat/completions");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    ctx.request_headers_to_set.push((
        http::HeaderName::from_static("anthropic-version"),
        http::HeaderValue::from_static("2026-10-21"),
    ));

    drop(filter.on_request(&mut ctx).await.unwrap());

    assert!(
        ctx.extra_request_headers.is_empty(),
        "native default must not override a queued contract"
    );
}

#[tokio::test]
async fn custom_default_version() {
    let filter = make_filter("default_version: \"2024-06-01\"");
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/messages");

    let req: &'static praxis_filter::Request = Box::leak(Box::new(req));
    let mut ctx = crate::test_utils::make_filter_context(req);

    drop(filter.on_request(&mut ctx).await.unwrap());

    let headers: std::collections::HashMap<&str, &str> = ctx
        .extra_request_headers
        .iter()
        .map(|(k, v)| (k.as_ref(), v.as_str()))
        .collect();

    assert_eq!(
        headers.get("anthropic-version"),
        Some(&"2024-06-01"),
        "should inject configured version"
    );
}

// -----------------------------------------------------------------------------
// MCP Connector Gate
// -----------------------------------------------------------------------------

/// A create-message body that uses the MCP connector.
const MCP_BODY: &[u8] = br#"{"model":"claude-opus-4-8","max_tokens":64,"messages":[{"role":"user","content":"Hi"}],"mcp_servers":[{"type":"url","url":"https://mcp.example.com/sse","name":"example"}],"tools":[{"type":"mcp_toolset","mcp_server_name":"example"}]}"#;

/// A create-message body with no MCP connector.
const PLAIN_BODY: &[u8] = br#"{"model":"claude-opus-4-8","max_tokens":64,"messages":[{"role":"user","content":"Hi"}]}"#;

#[tokio::test]
async fn rejects_mcp_connector_for_a_non_anthropic_provider() {
    let action = run_pipeline(Some("vllm"), MCP_BODY).await;
    let FilterAction::Reject(rejection) = action else {
        panic!("the MCP connector must be refused for a vLLM cluster");
    };
    assert_eq!(rejection.status, 400);
    let parsed: serde_json::Value = serde_json::from_slice(rejection.body.as_deref().unwrap()).unwrap();
    assert_eq!(parsed["type"], "error");
    assert_eq!(parsed["error"]["type"], "invalid_request_error");
    assert!(
        parsed["error"]["message"].as_str().unwrap().contains("MCP connector"),
        "the message must name the feature: {parsed}"
    );
}

#[tokio::test]
async fn forwards_mcp_connector_to_an_anthropic_provider() {
    assert!(
        matches!(run_pipeline(Some("anthropic"), MCP_BODY).await, FilterAction::Continue),
        "Anthropic executes the connector, so the request must pass"
    );
}

#[tokio::test]
async fn forwards_mcp_connector_to_an_untagged_cluster() {
    assert!(
        matches!(run_pipeline(None, MCP_BODY).await, FilterAction::Continue),
        "an untagged cluster may be Anthropic, so the request must pass"
    );
}

#[tokio::test]
async fn forwards_requests_without_the_connector_to_any_provider() {
    assert!(
        matches!(run_pipeline(Some("vllm"), PLAIN_BODY).await, FilterAction::Continue),
        "only the MCP connector is refused"
    );
}

#[tokio::test]
async fn placement_before_selection_does_not_reject() {
    let yaml = r#"
- filter: anthropic_messages_request
- filter: anthropic_messages_protocol
- filter: router
  routes:
    - path: /v1/messages
      cluster: backend
- filter: load_balancer
  clusters:
    - name: backend
      http:
        application_provider: vllm
      endpoints:
        - "127.0.0.1:8000"
"#;
    assert!(
        matches!(execute(yaml, MCP_BODY).await, FilterAction::Continue),
        "before load balancing no provider is selected, so the gate cannot apply"
    );
}

// -----------------------------------------------------------------------------
// Test Utilities
// -----------------------------------------------------------------------------

/// Build a filter from a YAML snippet.
fn make_filter(yaml_str: &str) -> Box<dyn HttpFilter> {
    let yaml: serde_yaml::Value = serde_yaml::from_str(yaml_str).unwrap();
    AnthropicMessagesProtocolFilter::from_config(&yaml).unwrap()
}

/// Run `body` through the shipped native order: request processing, routing,
/// load balancing onto a cluster tagged with `provider`, then this filter.
async fn run_pipeline(provider: Option<&str>, body: &'static [u8]) -> FilterAction {
    let http = provider.map_or_else(String::new, |provider| {
        format!("      http:\n        application_provider: {provider}\n")
    });
    let yaml = format!(
        r#"
- filter: anthropic_messages_request
- filter: router
  routes:
    - path: /v1/messages
      cluster: backend
- filter: load_balancer
  clusters:
    - name: backend
{http}      endpoints:
        - "127.0.0.1:8000"
- filter: anthropic_messages_protocol
"#
    );
    execute(&yaml, body).await
}

/// Build a pipeline from `yaml` and drive one `POST /v1/messages` through the
/// body pre-read and the request phase, as the proxy does.
async fn execute(yaml: &str, body: &'static [u8]) -> FilterAction {
    let mut registry = praxis_filter::FilterRegistry::with_builtins();
    praxis_filter::register_filters!(
        @register registry,
        http "anthropic_messages_request" => crate::anthropic::AnthropicMessagesRequestFilter::from_config
    );
    praxis_filter::register_filters!(
        @register registry,
        http "anthropic_messages_protocol" => AnthropicMessagesProtocolFilter::from_config
    );
    let mut entries: Vec<praxis_filter::FilterEntry> = serde_yaml::from_str(yaml).unwrap();
    let pipeline = praxis_filter::FilterPipeline::build(&mut entries, &registry).unwrap();

    let req = crate::test_utils::make_request(http::Method::POST, "/v1/messages");
    let req: &'static praxis_filter::Request = Box::leak(Box::new(req));
    let mut ctx = crate::test_utils::make_filter_context(req);
    let mut bytes = Some(bytes::Bytes::from_static(body));
    let body_action = pipeline
        .execute_http_request_body(&mut ctx, &mut bytes, true)
        .await
        .unwrap();
    assert!(
        !matches!(body_action, FilterAction::Reject(_)),
        "the body pre-read must accept a well-formed create-message body"
    );
    pipeline.execute_http_request(&mut ctx).await.unwrap()
}
