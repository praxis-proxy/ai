// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Tests for the token_count filter example configuration.
//!
//! The filter writes to `filter_metadata` which is not observable from
//! an HTTP response, so most cases only verify the proxy starts and
//! proxies traffic correctly. Token extraction correctness is covered
//! by unit tests in `praxis-ai-filters`.
//!
//! The Bedrock `InvokeModel` header path is the exception: it is
//! observable end-to-end by chaining `token_count` with
//! `token_usage_headers`, which surfaces the extracted counts as
//! response headers.
//!
//! Buffered JSON extraction is observed after the body phase by a test-only
//! filter, using Azure's real response adapter to drive buffered delivery.

use std::{
    collections::HashMap,
    sync::{Arc, mpsc},
    time::Duration,
};

use bytes::Bytes;
use praxis_core::config::Config;
use praxis_filter::{BodyAccess, BodyMode, FilterAction, FilterError, FilterFactory, HttpFilter, HttpFilterContext};
use praxis_test_utils::{
    Backend, free_port, http_send, json_post, parse_body, parse_header, parse_status, start_backend_with_shutdown,
    start_proxy, start_proxy_with_registry, test_subrequest_client,
};

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[test]
fn token_count_proxies_response() {
    let backend_port_guard = start_backend_with_shutdown("ok");
    let backend_port = backend_port_guard.port();
    let proxy_port = free_port();
    let config = super::load_example_config(
        "token-counting.yaml",
        proxy_port,
        HashMap::from([("127.0.0.1:3000", backend_port)]),
    );
    let proxy = start_proxy(&config);

    let raw = http_send(
        proxy.addr(),
        "GET / HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n",
    );
    assert_eq!(parse_status(&raw), 200, "proxy should return 200");
}

#[test]
fn token_count_bedrock_invoke_model_extracts_header_counts() {
    let backend_port_guard = Backend::fixed("ok")
        .header("x-amzn-bedrock-input-token-count", "25")
        .header("x-amzn-bedrock-output-token-count", "50")
        .start_with_shutdown();
    let backend_port = backend_port_guard.port();
    let proxy_port = free_port();
    let config = Config::from_yaml(&make_bedrock_invoke_model_yaml(proxy_port, backend_port)).unwrap();
    let proxy = start_proxy(&config);

    let raw = http_send(
        proxy.addr(),
        "GET / HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n",
    );

    assert_eq!(parse_status(&raw), 200, "proxy should return 200");
    assert_eq!(
        parse_header(&raw, "praxis-token-input"),
        Some("25".to_owned()),
        "input token count should be extracted from Bedrock InvokeModel headers"
    );
    assert_eq!(
        parse_header(&raw, "praxis-token-output"),
        Some("50".to_owned()),
        "output token count should be extracted from Bedrock InvokeModel headers"
    );
    assert_eq!(
        parse_header(&raw, "praxis-token-total"),
        Some("75".to_owned()),
        "total token count should be computed from input + output"
    );
}

#[test]
fn token_count_extracts_buffered_azure_json_once() {
    let backend = Backend::chunked(vec![
        r#"{"choices":[],"prompt_filter_results":[],"usage":{"prompt_tokens":15,"#.to_owned(),
        r#""completion_tokens":42,"total_tokens":57}}"#.to_owned(),
    ])
    .header("content-type", "application/json")
    .stall_after_first_chunk(Duration::from_millis(50))
    .start_with_shutdown();
    let config = Config::from_yaml(&make_buffered_azure_yaml(free_port(), backend.port())).unwrap();
    let (sender, receiver) = mpsc::channel();
    let mut registry = praxis_ai::build_full_registry(&test_subrequest_client());
    registry
        .register(
            "test_token_observer",
            FilterFactory::Http(Arc::new(move |_| Ok(Box::new(TokenObserver(sender.clone()))))),
        )
        .unwrap();
    let proxy = start_proxy_with_registry(&config, &registry);

    let raw = http_send(proxy.addr(), &json_post("/v1/chat/completions", "{}"));
    assert_eq!(parse_status(&raw), 200, "the Azure response should succeed");
    let body: serde_json::Value = serde_json::from_str(&parse_body(&raw)).unwrap();
    assert!(
        body.get("prompt_filter_results").is_none(),
        "Azure fields should be removed"
    );
    assert_eq!(body["usage"]["total_tokens"], 57, "the adapter should preserve usage");

    let (counts, mode, saw_raw) = receiver.recv_timeout(Duration::from_secs(5)).unwrap();
    assert!(
        matches!(mode, BodyMode::StreamBuffer { .. }),
        "Azure should enable buffering"
    );
    assert!(saw_raw, "raw chunks must precede the frozen EOS delivery");
    assert_eq!(
        counts,
        [
            Some("15".to_owned()),
            Some("42".to_owned()),
            Some("57".to_owned()),
            None
        ],
        "buffered JSON should publish exact usage without overflow"
    );
}

// -----------------------------------------------------------------------------
// Test Utilities
// -----------------------------------------------------------------------------

/// Body-phase observation includes the overflow signal and buffered delivery.
type TokenObservation = ([Option<String>; 4], BodyMode, bool);

/// Observes metadata once headers are already committed and cannot expose it.
struct TokenObserver(mpsc::Sender<TokenObservation>);

#[async_trait::async_trait]
impl HttpFilter for TokenObserver {
    fn name(&self) -> &'static str {
        "test_token_observer"
    }

    fn response_body_access(&self) -> BodyAccess {
        BodyAccess::ReadOnly
    }

    async fn on_request(&self, _ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        Ok(FilterAction::Continue)
    }

    fn on_response_body(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        _body: &mut Option<Bytes>,
        end_of_stream: bool,
    ) -> Result<FilterAction, FilterError> {
        if end_of_stream {
            let counts = ["token.input", "token.output", "token.total", "token.status"]
                .map(|key| ctx.get_metadata(key).map(str::to_owned));
            self.0
                .send((
                    counts,
                    ctx.response_body_mode,
                    ctx.get_metadata("test.token.raw_seen").is_some(),
                ))
                .unwrap();
        } else {
            ctx.set_metadata("test.token.raw_seen", "true");
        }
        Ok(FilterAction::Continue)
    }
}

/// Reverse response-body order runs the Azure adapter, counter, then observer.
fn make_buffered_azure_yaml(proxy_port: u16, backend_port: u16) -> String {
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
      - filter: test_token_observer
      - filter: token_count
        provider: azure
      - filter: openai_chat_completions_to_azureai_chat_completions
      - filter: load_balancer
        clusters:
          - name: backend
            endpoints:
              - "127.0.0.1:{backend_port}"
insecure_options:
  allow_private_endpoints: true
"#
    )
}

/// Builds a config chaining `token_count` (Bedrock `InvokeModel`) with
/// `token_usage_headers`, so the extracted counts become observable as
/// response headers.
///
/// Response hooks run in *reverse* declared order (last filter's
/// `on_response` fires first), so `token_usage_headers` is declared
/// before `token_count` here to ensure `token_count` populates
/// `filter_metadata` before `token_usage_headers` reads it.
fn make_bedrock_invoke_model_yaml(proxy_port: u16, backend_port: u16) -> String {
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
      - filter: token_usage_headers
      - filter: token_count
        provider: bedrock_invoke_model
      - filter: load_balancer
        clusters:
          - name: backend
            endpoints:
              - "127.0.0.1:{backend_port}"
insecure_options:
  allow_private_endpoints: true
"#
    )
}
