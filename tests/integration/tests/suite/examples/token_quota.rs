// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Functional coverage for PPE-driven token quota.

use std::{collections::HashMap, sync::mpsc, time::Duration};

use async_trait::async_trait;
use bytes::Bytes;
use praxis_core::config::{Config, FailureMode, FilterEntry};
use praxis_filter::{
    BodyAccess, BodyMode, FilterAction, FilterError, FilterFactory, HttpFilter, HttpFilterContext, PolicyResponseOutput,
};
#[cfg(feature = "token-rate-limit-filter")]
use praxis_test_utils::start_proxy;
use praxis_test_utils::{
    Backend, StatefulCapturingBackend, free_port, http_send, json_post, parse_body, parse_status,
    start_proxy_with_registry, test_subrequest_client,
};

/// The shipped example evaluates PPE quota and leaves the provider response unchanged.
#[test]
fn example_config_token_quota() {
    let response = r#"{"choices":[],"usage":{"prompt_tokens":120,"completion_tokens":40,"total_tokens":160,"prompt_tokens_details":{"cached_tokens":90}}}"#;
    let backend = StatefulCapturingBackend::new(vec![(200, response.to_owned())]).start_with_shutdown();
    let mut config = super::load_example_config(
        "token-quota.yaml",
        free_port(),
        HashMap::from([("127.0.0.1:3000", backend.port())]),
    );
    let policy_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../examples/configs/token-quota.ppe.yaml")
        .canonicalize()
        .expect("example policy path");
    let policy = config.filter_chains[0]
        .filters
        .iter_mut()
        .find(|entry| entry.filter_type == "policy")
        .expect("example policy filter");
    policy.config.as_mapping_mut().expect("policy config mapping").insert(
        serde_yaml::Value::String("config_path".to_owned()),
        serde_yaml::Value::String(policy_path.to_string_lossy().into_owned()),
    );
    config.filter_chains[0].filters.insert(
        2,
        FilterEntry {
            filter_type: "test_quota_observer".to_owned(),
            branch_chains: None,
            conditions: Vec::new(),
            name: None,
            response_conditions: Vec::new(),
            failure_mode: FailureMode::default(),
            config: serde_yaml::Value::Null,
        },
    );
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
    let requests = backend.requests();
    let forwarded = requests.first().expect("provider should receive one request");
    assert_eq!(requests.len(), 1, "provider should receive exactly one request");
    assert_eq!(forwarded.method, "POST");
    assert_eq!(forwarded.uri, "/v1/chat/completions");
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&forwarded.body).expect("forwarded body should be JSON"),
        serde_json::json!({
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "hi"}]
        }),
        "quota filters must preserve the provider request"
    );
    assert_eq!(
        receiver
            .recv_timeout(Duration::from_secs(5))
            .expect("observer should receive PPE quota output"),
        [
            Some("53".to_owned()),
            Some("weighted-quota-v1".to_owned()),
            Some("quota_units".to_owned()),
            Some("evaluated".to_owned()),
        ]
    );
}

/// The final quota value and its audit fields are reusable by a downstream filter.
#[test]
fn weighted_quota_is_written_back_after_token_count() {
    let response = r#"{"choices":[],"usage":{"prompt_tokens":120,"completion_tokens":40,"total_tokens":160,"prompt_tokens_details":{"cached_tokens":90,"cache_write_tokens":20}}}"#;
    let backend = Backend::fixed(response)
        .header("content-type", "application/json")
        .start_with_shutdown();
    let (_policy, yaml) = config_yaml(free_port(), backend.port());
    let config = Config::from_yaml(&yaml).expect("config should parse");
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
        .expect("observer should receive PPE quota output");
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
    let (_policy, yaml) = config_yaml(free_port(), backend.port());
    let config = Config::from_yaml(&yaml).expect("config should parse");
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
        .expect("observer should receive final streaming PPE quota output");
    assert_eq!(observed[0].as_deref(), Some("17"));
    assert_eq!(observed[3].as_deref(), Some("evaluated"));
}

/// The limiter consumes the shared result instead of recomputing raw-token weights.
#[cfg(feature = "token-rate-limit-filter")]
#[test]
fn token_rate_limit_reconciles_from_shared_quota_output() {
    // Two unreconciled 50-token reservations (100) would exceed the 90-unit
    // window. The formula yields ceil(10/3 + 10 + 90/30) = 17, so a second
    // reservation fits only when the limiter settles from the quota result.
    let response = r#"{"choices":[],"usage":{"prompt_tokens":100,"completion_tokens":10,"total_tokens":110,"prompt_tokens_details":{"cached_tokens":90}}}"#;
    let backend = Backend::fixed(response)
        .header("content-type", "application/json")
        .start_with_shutdown();
    let (_policy, yaml) = rate_limit_config_yaml(free_port(), backend.port());
    let config = Config::from_yaml(&yaml).expect("config should parse");
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

/// Test-only downstream consumer proving PPE output ordering.
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
            let quota = ctx
                .extensions
                .get::<PolicyResponseOutput>()
                .and_then(|output| output.get("token_quota"));
            let values = quota.map_or([None, None, None, None], |quota| {
                [
                    quota
                        .get("value")
                        .and_then(serde_json::Value::as_u64)
                        .map(|value| value.to_string()),
                    quota
                        .get("formula_id")
                        .and_then(serde_json::Value::as_str)
                        .map(str::to_owned),
                    quota.get("unit").and_then(serde_json::Value::as_str).map(str::to_owned),
                    quota
                        .get("status")
                        .and_then(serde_json::Value::as_str)
                        .map(str::to_owned),
                ]
            });
            self.sender
                .send(values)
                .map_err(|error| FilterError::from(error.to_string()))?;
        }
        Ok(FilterAction::Continue)
    }
}

/// Build a real response chain: token_count -> policy/PPE -> observer.
fn config_yaml(proxy_port: u16, backend_port: u16) -> (tempfile::TempDir, String) {
    let (policy, policy_path) = write_policy();
    let yaml = format!(
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
      - filter: policy
        config_path: "{policy_path}"
        body_access: read_only
      - filter: token_count
        provider: openai
      - filter: load_balancer
        clusters:
          - name: backend
            endpoints: ["127.0.0.1:{backend_port}"]
insecure_options:
  allow_private_endpoints: true
"#
    );
    (policy, yaml)
}

/// Build an integrated limiter -> quota -> counter response chain.
#[cfg(feature = "token-rate-limit-filter")]
fn rate_limit_config_yaml(proxy_port: u16, backend_port: u16) -> (tempfile::TempDir, String) {
    let (policy, policy_path) = write_policy();
    let yaml = format!(
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
            capacity: 90
            reserved_tokens: 50
      - filter: policy
        config_path: "{policy_path}"
        body_access: read_only
      - filter: token_count
        provider: openai
      - filter: load_balancer
        clusters:
          - name: backend
            endpoints: ["127.0.0.1:{backend_port}"]
insecure_options:
  allow_private_endpoints: true
"#
    );
    (policy, yaml)
}

/// Write the PPE document that selects the weighted quota plugin.
fn write_policy() -> (tempfile::TempDir, String) {
    let dir = tempfile::TempDir::new().expect("policy tempdir");
    let path = dir.path().join("token-quota.ppe.yaml");
    std::fs::write(
        &path,
        r#"plugins:
  - name: weighted-quota-v1
    kind: ai/token-quota
    hooks: [cmf.llm_output]
    capabilities: [metadata_only_response]
    mode: sequential
    on_error: fail
    config:
      formula_id: weighted-quota-v1
      unit: quota_units
      weights:
        input: 0.3333333333333333
        cached_input: 0.03333333333333333
        cache_write: 0.4166666666666667
        output: 1.0
        reasoning: 1.0
routes:
  - llm: "*"
    authorization:
      post_invocation:
        - "run(weighted-quota-v1)"
"#,
    )
    .expect("write policy");
    let path = path.to_str().expect("utf8 policy path").to_owned();
    (dir, path)
}
