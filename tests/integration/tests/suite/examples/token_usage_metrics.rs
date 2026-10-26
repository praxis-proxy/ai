// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Integration tests for the `token_usage_metrics` example config.
//!
//! Verifies transparent proxying and that unverified client identity cannot
//! create a tenant Prometheus series.

use std::collections::HashMap;

use async_trait::async_trait;
use praxis_core::config::Config;
use praxis_filter::{FilterAction, FilterError, HttpFilter, HttpFilterContext};
use praxis_test_utils::{
    Backend, free_port, http_send, load_example_config, parse_body, parse_status, registry_with, start_proxy,
    start_proxy_with_registry, wait_for_http,
};

const OPENAI_JSON: &str = r#"{"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":20,"total_tokens":30}}"#;

fn json_post_with_headers(path: &str, body: &str, headers: &[(&str, &str)]) -> String {
    let mut extra = String::new();
    for (name, value) in headers {
        extra.push_str(&format!("{name}: {value}\r\n"));
    }
    format!(
        "POST {path} HTTP/1.1\r\n\
         Host: localhost\r\n\
         Content-Type: application/json\r\n\
         Content-Length: {}\r\n\
         {extra}\
         Connection: close\r\n\r\n\
         {body}",
        body.len()
    )
}

#[test]
fn example_config_rejects_unverified_tenant_metrics() {
    let backend = Backend::fixed(OPENAI_JSON)
        .header("content-type", "application/json")
        .start_with_shutdown();
    let proxy_port = free_port();
    let admin_port = free_port();
    let mut config = load_example_config(
        "token-usage-metrics.yaml",
        proxy_port,
        HashMap::from([("127.0.0.1:3000", backend.port())]),
    );
    let admin_addr = format!("127.0.0.1:{admin_port}");
    config.admin.metrics_address = Some(admin_addr.clone());
    let proxy = start_proxy(&config);
    wait_for_http(&admin_addr);

    let raw = http_send(
        proxy.addr(),
        &json_post_with_headers(
            "/v1/chat/completions",
            r#"{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}"#,
            &[("x-tenant-username", "token-metrics-example-alice")],
        ),
    );
    assert_eq!(parse_status(&raw), 200, "example config smoke test should return 200");
    assert_eq!(parse_body(&raw), OPENAI_JSON, "body should pass through unchanged");

    let metrics = http_send(
        &admin_addr,
        "GET /metrics HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n",
    );
    assert_eq!(parse_status(&metrics), 200, "metrics endpoint should return 200");
    let metrics_body = parse_body(&metrics);
    assert!(
        !metrics_body.contains("tenant=\"token-metrics-example-alice\""),
        "raw client identity must not create a tenant metric series: {metrics_body}"
    );
}

/// Test-only authentication stand-in: production deployments must populate
/// this metadata from verified credentials, never from a client-controlled header.
struct TrustedTenantMetadata;

#[async_trait]
impl HttpFilter for TrustedTenantMetadata {
    fn name(&self) -> &'static str {
        "trusted_tenant_metadata"
    }

    async fn on_request(&self, ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        ctx.set_metadata("x-tenant-username", "token-metrics-example-alice");
        Ok(FilterAction::Continue)
    }
}

#[test]
fn example_config_emits_metrics_for_authenticated_tenant() {
    let backend = Backend::fixed(OPENAI_JSON)
        .header("content-type", "application/json")
        .start_with_shutdown();
    let proxy_port = free_port();
    let admin_port = free_port();
    let yaml = format!(
        r#"
listeners:
  - name: default
    address: "127.0.0.1:{proxy_port}"
    filter_chains: [main]
filter_chains:
  - name: main
    filters:
      - filter: trusted_tenant_metadata
      - filter: router
        routes:
          - path_prefix: "/"
            cluster: backend
      - filter: token_usage_metrics
        tenant_metadata_key: "x-tenant-username"
        allowed_tenants: [token-metrics-example-alice]
        allowed_models: [gpt-4o]
      - filter: token_count
        provider: openai
      - filter: load_balancer
        clusters:
          - name: backend
            endpoints: ["127.0.0.1:{backend_port}"]
admin:
  metrics_address: "127.0.0.1:{admin_port}"
insecure_options:
  allow_private_endpoints: true
"#,
        backend_port = backend.port(),
    );
    let config = Config::from_yaml(&yaml).expect("authenticated metrics config should parse");
    let registry = registry_with("trusted_tenant_metadata", || Box::new(TrustedTenantMetadata));
    let proxy = start_proxy_with_registry(&config, &registry);
    let admin_addr = format!("127.0.0.1:{admin_port}");
    wait_for_http(&admin_addr);

    let raw = http_send(
        proxy.addr(),
        &json_post_with_headers(
            "/v1/chat/completions",
            r#"{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}"#,
            &[],
        ),
    );
    assert_eq!(parse_status(&raw), 200, "authenticated example should return 200");
    assert_eq!(parse_body(&raw), OPENAI_JSON, "body should pass through unchanged");

    let metrics = http_send(
        &admin_addr,
        "GET /metrics HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n",
    );
    let metrics_body = parse_body(&metrics);
    for kind in ["input", "output", "total"] {
        assert!(
            metrics_body.lines().any(|line| {
                line.starts_with("praxis_ai_tokens_total{")
                    && line.contains("tenant=\"token-metrics-example-alice\"")
                    && line.contains("model=\"gpt-4o\"")
                    && line.contains(&format!("kind=\"{kind}\""))
            }),
            "authenticated token metric for {kind} should be exported: {metrics_body}"
        );
    }
}
