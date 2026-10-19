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
    io::{BufRead as _, BufReader, Write as _},
    net::TcpListener,
    sync::{Arc, mpsc},
    thread::{self, JoinHandle},
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

/// The real Azure adapter delivers raw chunks followed by one frozen JSON body.
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
            FilterFactory::Http(Arc::new(move |_| {
                Ok(Box::new(TokenObserver {
                    observations: sender.clone(),
                    release: None,
                }))
            })),
        )
        .unwrap();
    let proxy = start_proxy_with_registry(&config, &registry);

    let raw = http_send(proxy.addr(), &json_post("/v1/chat/completions", "{}"));
    assert_eq!(parse_status(&raw), 200, "the Azure response should succeed");
    let body: serde_json::Value = serde_json::from_str(&parse_body(&raw)).unwrap();
    let expected_body = serde_json::json!({
        "choices": [],
        "usage": {
            "prompt_tokens": 15,
            "completion_tokens": 42,
            "total_tokens": 57
        }
    });
    assert_eq!(body, expected_body, "the adapter should return the complete response");

    let (counts, mode, saw_raw, final_body) = receiver.recv_timeout(Duration::from_secs(5)).unwrap();
    let frozen_body: serde_json::Value =
        serde_json::from_slice(final_body.as_deref().expect("Azure EOS must contain frozen JSON")).unwrap();
    assert_eq!(
        frozen_body, expected_body,
        "Azure EOS should contain the complete adapted JSON"
    );
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

/// A real release preserves the captured prefix when EOS contains only the tail.
#[test]
fn token_count_extracts_json_after_response_release() {
    let (released_sender, released_receiver) = mpsc::channel();
    let (backend_port, backend_thread) = start_released_json_backend(released_receiver);
    let config = Config::from_yaml(&make_token_count_yaml(
        free_port(),
        backend_port,
        "openai",
        "test_token_observer",
    ))
    .unwrap();
    let (sender, receiver) = mpsc::channel();
    let mut registry = praxis_ai::build_full_registry(&test_subrequest_client());
    registry
        .register(
            "test_token_observer",
            FilterFactory::Http(Arc::new(move |_| {
                Ok(Box::new(TokenObserver {
                    observations: sender.clone(),
                    release: Some(released_sender.clone()),
                }))
            })),
        )
        .unwrap();
    let proxy = start_proxy_with_registry(&config, &registry);

    let raw = http_send(
        proxy.addr(),
        "GET / HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n",
    );
    backend_thread.join().unwrap();
    assert_eq!(parse_status(&raw), 200, "the released response should succeed");
    assert_eq!(
        parse_body(&raw),
        format!("{RELEASED_JSON_PREFIX}{RELEASED_JSON_TAIL}"),
        "the client should receive the prefix and tail exactly once"
    );
    let (counts, mode, saw_raw, final_body) = receiver.recv_timeout(Duration::from_secs(5)).unwrap();
    assert!(saw_raw, "a raw prefix must trigger the real Release action");
    assert!(
        matches!(mode, BodyMode::StreamBuffer { .. }),
        "the pinned 0.7.4 runtime retains StreamBuffer after Release"
    );
    assert_eq!(
        final_body.as_deref(),
        Some(RELEASED_JSON_TAIL.as_bytes()),
        "EOS must deliver only the final tail after Release"
    );
    assert_eq!(
        counts,
        [Some("7".to_owned()), Some("11".to_owned()), Some("18".to_owned()), None],
        "released JSON should publish exact usage without overflow"
    );
}

// -----------------------------------------------------------------------------
// Test Utilities
// -----------------------------------------------------------------------------

/// Incomplete JSON sent before the observer releases the response buffer.
const RELEASED_JSON_PREFIX: &str = r#"{"choices":[],"usage":{"prompt_tokens":7,"#;

/// Final JSON bytes withheld until the response filter requests release.
const RELEASED_JSON_TAIL: &str = r#""completion_tokens":11,"total_tokens":18}}"#;

/// Body-phase observation includes overflow, delivery mode, and the EOS payload.
type TokenObservation = ([Option<String>; 4], BodyMode, bool, Option<Bytes>);

/// Observes metadata once headers are already committed and cannot expose it.
struct TokenObserver {
    /// Receives token counts after the counter handles the EOS callback.
    observations: mpsc::Sender<TokenObservation>,
    /// When present, requests buffering and releases the first raw chunk.
    release: Option<mpsc::Sender<()>>,
}

#[async_trait::async_trait]
impl HttpFilter for TokenObserver {
    fn name(&self) -> &'static str {
        "test_token_observer"
    }

    fn response_body_access(&self) -> BodyAccess {
        BodyAccess::ReadOnly
    }

    fn response_body_mode(&self) -> BodyMode {
        self.release
            .as_ref()
            .map_or(BodyMode::Stream, |_| BodyMode::StreamBuffer {
                max_bytes: Some(65_536),
            })
    }

    async fn on_request(&self, _ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        Ok(FilterAction::Continue)
    }

    fn on_response_body(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        body: &mut Option<Bytes>,
        end_of_stream: bool,
    ) -> Result<FilterAction, FilterError> {
        if end_of_stream {
            let counts = ["token.input", "token.output", "token.total", "token.status"]
                .map(|key| ctx.get_metadata(key).map(str::to_owned));
            self.observations
                .send((
                    counts,
                    ctx.response_body_mode,
                    ctx.get_metadata("test.token.raw_seen").is_some(),
                    body.clone(),
                ))
                .unwrap();
        } else {
            let first_chunk = ctx.get_metadata("test.token.raw_seen").is_none();
            ctx.set_metadata("test.token.raw_seen", "true");
            if first_chunk
                && body.is_some()
                && let Some(release) = &self.release
            {
                release.send(()).unwrap();
                return Ok(FilterAction::Release);
            }
        }
        Ok(FilterAction::Continue)
    }
}

/// A fixed-length backend waits for the real release callback before sending EOS.
fn start_released_json_backend(released: mpsc::Receiver<()>) -> (u16, JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let handle = thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        stream.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
        let mut reader = BufReader::new(&mut stream);
        let mut line = String::new();
        while reader.read_line(&mut line).unwrap() > 0 && line != "\r\n" {
            line.clear();
        }
        let body_len = RELEASED_JSON_PREFIX
            .len()
            .checked_add(RELEASED_JSON_TAIL.len())
            .unwrap();
        write!(
            stream,
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {body_len}\r\nConnection: close\r\n\r\n{RELEASED_JSON_PREFIX}"
        )
        .unwrap();
        stream.flush().unwrap();
        released.recv_timeout(Duration::from_secs(5)).unwrap();
        stream.write_all(RELEASED_JSON_TAIL.as_bytes()).unwrap();
    });
    (port, handle)
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
    make_token_count_yaml(proxy_port, backend_port, "bedrock_invoke_model", "token_usage_headers")
}

/// Response hooks run the counter before the observer in reverse pipeline order.
fn make_token_count_yaml(proxy_port: u16, backend_port: u16, provider: &str, observer: &str) -> String {
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
      - filter: {observer}
      - filter: token_count
        provider: {provider}
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
