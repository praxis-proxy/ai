// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Functional coverage for the `token_quota` CEL bridge.

use std::{collections::HashMap, sync::mpsc, time::Duration};

use async_trait::async_trait;
use bytes::Bytes;
use praxis_core::config::Config;
use praxis_filter::{BodyAccess, BodyMode, FilterAction, FilterError, FilterFactory, HttpFilter, HttpFilterContext};
use praxis_test_utils::{
    Backend, free_port, http_send, json_post, parse_body, parse_status, start_proxy, start_proxy_with_registry,
    test_subrequest_client,
};

/// The shipped example starts and leaves the provider response unchanged.
#[test]
fn example_config_token_quota() {
    let response = r#"{"choices":[],"usage":{"prompt_tokens":120,"completion_tokens":40,"total_tokens":160,"prompt_tokens_details":{"cached_tokens":90}}}"#;
    let backend = Backend::fixed(response)
        .header("content-type", "application/json")
        .start_with_shutdown();
    let config = super::load_example_config(
        "token-quota.yaml",
        free_port(),
        HashMap::from([("127.0.0.1:3000", backend.port())]),
    );
    let proxy = start_proxy(&config);
    let raw = http_send(
        proxy.addr(),
        &json_post(
            "/v1/chat/completions",
            r#"{"model":"gpt-4","messages":[{"role":"user","content":"hi"}]}"#,
        ),
    );
    assert_eq!(parse_status(&raw), 200);
    assert_eq!(
        parse_body(&raw),
        response,
        "quota evaluation must not mutate the response"
    );
}

/// The final quota value and its audit metadata are reusable by a downstream filter.
#[test]
fn weighted_quota_is_written_back_after_token_count() {
    let response = r#"{"choices":[],"usage":{"prompt_tokens":120,"completion_tokens":40,"total_tokens":160,"prompt_tokens_details":{"cached_tokens":90,"cache_write_tokens":20}}}"#;
    let backend = Backend::fixed(response)
        .header("content-type", "application/json")
        .start_with_shutdown();
    let config = Config::from_yaml(&config_yaml(free_port(), backend.port())).expect("config should parse");
    let (sender, receiver) = mpsc::channel();
    let mut registry = praxis_ai::build_full_registry(&test_subrequest_client());
    registry
        .register(
            "test_quota_observer",
            FilterFactory::Http(std::sync::Arc::new(move |_| {
                Ok(Box::new(QuotaObserver { sender: sender.clone() }))
            })),
        )
        .expect("observer registration should succeed");
    let proxy = start_proxy_with_registry(&config, &registry);
    let raw = http_send(proxy.addr(), &json_post("/v1/chat/completions", r#"{"model":"gpt-4"}"#));
    assert_eq!(parse_status(&raw), 200);

    let observed = receiver
        .recv_timeout(Duration::from_secs(5))
        .expect("observer should receive quota metadata");
    assert_eq!(
        observed,
        [
            Some("55".to_owned()),
            Some("weighted-quota-v1".to_owned()),
            Some("quota_units".to_owned()),
            Some("evaluated".to_owned()),
        ]
    );
}

/// Split SSE usage is evaluated once from the final normalized snapshot.
#[test]
fn streaming_usage_is_evaluated_at_end_of_stream() {
    let chunks = vec![
        "data: {\"choices\":[{\"delta\":{\"content\":\"Hi\"}}],\"usage\":null}\n\n".to_owned(),
        "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":100,".to_owned(),
        "\"completion_tokens\":10,\"total_tokens\":110,\"prompt_tokens_details\":{\"cached_tokens\":90}}}\n\n"
            .to_owned(),
        "data: [DONE]\n\n".to_owned(),
    ];
    let expected_body = chunks.concat();
    let backend = Backend::chunked(chunks)
        .header("content-type", "text/event-stream")
        .header("cache-control", "no-cache")
        .start_with_shutdown();
    let config = Config::from_yaml(&config_yaml(free_port(), backend.port())).expect("config should parse");
    let (sender, receiver) = mpsc::channel();
    let mut registry = praxis_ai::build_full_registry(&test_subrequest_client());
    registry
        .register(
            "test_quota_observer",
            FilterFactory::Http(std::sync::Arc::new(move |_| {
                Ok(Box::new(QuotaObserver { sender: sender.clone() }))
            })),
        )
        .expect("observer registration should succeed");
    let proxy = start_proxy_with_registry(&config, &registry);
    let raw = http_send(
        proxy.addr(),
        &json_post("/v1/chat/completions", r#"{"model":"gpt-4","stream":true}"#),
    );
    assert_eq!(parse_status(&raw), 200);
    assert_eq!(parse_body(&raw), expected_body);

    let observed = receiver
        .recv_timeout(Duration::from_secs(5))
        .expect("observer should receive final streaming quota metadata");
    assert_eq!(observed[0].as_deref(), Some("17"));
    assert_eq!(observed[3].as_deref(), Some("evaluated"));
}

/// The limiter consumes the shared result instead of recomputing raw-token weights.
#[cfg(feature = "token-rate-limit-filter")]
#[test]
fn token_rate_limit_reconciles_from_shared_quota_metadata() {
    // Raw total is 110, which would exhaust the 100-unit window. The formula
    // yields ceil(10/3 + 10 + 90/30) = 17, so a second reservation fits.
    let response = r#"{"choices":[],"usage":{"prompt_tokens":100,"completion_tokens":10,"total_tokens":110,"prompt_tokens_details":{"cached_tokens":90}}}"#;
    let backend = Backend::fixed(response)
        .header("content-type", "application/json")
        .start_with_shutdown();
    let config = Config::from_yaml(&rate_limit_config_yaml(free_port(), backend.port())).expect("config should parse");
    let proxy = start_proxy(&config);

    for request_number in 1..=2 {
        let raw = http_send(proxy.addr(), &json_post("/v1/chat/completions", r#"{"model":"gpt-4"}"#));
        assert_eq!(
            parse_status(&raw),
            200,
            "request {request_number} should fit after shared-quota reconciliation"
        );
    }
}

/// Test-only downstream consumer proving metadata write-back ordering.
struct QuotaObserver {
    /// Sends the final bounded quota contract to the test thread.
    sender: mpsc::Sender<[Option<String>; 4]>,
}

#[async_trait]
impl HttpFilter for QuotaObserver {
    fn name(&self) -> &'static str {
        "test_quota_observer"
    }

    async fn on_request(&self, _ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        Ok(FilterAction::Continue)
    }

    fn response_body_access(&self) -> BodyAccess {
        BodyAccess::ReadOnly
    }

    fn response_body_mode(&self) -> BodyMode {
        BodyMode::Stream
    }

    fn on_response_body(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        _body: &mut Option<Bytes>,
        end_of_stream: bool,
    ) -> Result<FilterAction, FilterError> {
        if end_of_stream {
            let values = [
                ctx.get_metadata("token.quota.value").map(str::to_owned),
                ctx.get_metadata("token.quota.formula").map(str::to_owned),
                ctx.get_metadata("token.quota.unit").map(str::to_owned),
                ctx.get_metadata("token.quota.status").map(str::to_owned),
            ];
            self.sender
                .send(values)
                .map_err(|error| FilterError::from(error.to_string()))?;
        }
        Ok(FilterAction::Continue)
    }
}

/// Build a real response chain: token_count -> token_quota -> observer.
fn config_yaml(proxy_port: u16, backend_port: u16) -> String {
    format!(
        r#"
listeners:
  - name: default
    address: "127.0.0.1:{proxy_port}"
    filter_chains: [main]
filter_chains:
  - name: main
    filters:
      - filter: router
        routes:
          - path_prefix: "/"
            cluster: backend
      - filter: test_quota_observer
      - filter: token_quota
        formula_id: weighted-quota-v1
        unit: quota_units
        expression: "double(usage.input.uncached + usage.unclassified) / 3.0 + double(usage.output.visible) + double(usage.input.cache_write) / 2.4 + double(usage.input.cache_read) / 30.0 + double(usage.output.reasoning)"
      - filter: token_count
        provider: openai
      - filter: load_balancer
        clusters:
          - name: backend
            endpoints: ["127.0.0.1:{backend_port}"]
insecure_options:
  allow_private_endpoints: true
"#
    )
}

/// Build an integrated limiter -> quota -> counter response chain.
#[cfg(feature = "token-rate-limit-filter")]
fn rate_limit_config_yaml(proxy_port: u16, backend_port: u16) -> String {
    format!(
        r#"
listeners:
  - name: default
    address: "127.0.0.1:{proxy_port}"
    filter_chains: [main]
filter_chains:
  - name: main
    filters:
      - filter: router
        routes:
          - path_prefix: "/"
            cluster: backend
      - filter: token_rate_limit
        use_token_quota: true
        rules:
          - name: default
            algorithm: sliding_window
            window: 1h
            capacity: 100
            reserved_tokens: 50
      - filter: token_quota
        formula_id: weighted-quota-v1
        expression: "double(usage.input.uncached + usage.unclassified) / 3.0 + double(usage.output.visible) + double(usage.input.cache_write) / 2.4 + double(usage.input.cache_read) / 30.0 + double(usage.output.reasoning)"
      - filter: token_count
        provider: openai
      - filter: load_balancer
        clusters:
          - name: backend
            endpoints: ["127.0.0.1:{backend_port}"]
insecure_options:
  allow_private_endpoints: true
"#
    )
}
