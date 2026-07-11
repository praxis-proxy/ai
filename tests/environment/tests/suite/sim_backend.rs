// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Simulator-backed environment integration tests.
//!
//! Proves the AI gateway can route OpenAI-compatible traffic
//! through generic `ext_proc` + `endpoint_selector` to an
//! `llm-d-inference-sim` container backend.

use std::{
    net::SocketAddr,
    pin::Pin,
    sync::{
        Arc,
        atomic::{AtomicU32, Ordering},
    },
};

use async_trait::async_trait;
use praxis_ai_llmd_ext_proc::proto::envoy::service::ext_proc::v3::{
    BodyResponse, HeadersResponse, ProcessingRequest, ProcessingResponse,
    external_processor_server::{ExternalProcessor, ExternalProcessorServer},
    processing_request, processing_response,
};
use praxis_test_utils::{free_port, json_post, parse_body, start_simulator};
use tokio::sync::oneshot;
use tonic::transport::Server;

// -----------------------------------------------------------------------------
// Mock Processor
// -----------------------------------------------------------------------------

struct SimRoutingProcessor {
    destination: String,
    stream_count: Arc<AtomicU32>,
}

#[async_trait]
impl ExternalProcessor for SimRoutingProcessor {
    type ProcessStream = Pin<Box<dyn futures::Stream<Item = Result<ProcessingResponse, tonic::Status>> + Send>>;

    async fn process(
        &self,
        request: tonic::Request<tonic::Streaming<ProcessingRequest>>,
    ) -> Result<tonic::Response<Self::ProcessStream>, tonic::Status> {
        self.stream_count.fetch_add(1, Ordering::Relaxed);
        let dest = self.destination.clone();
        let mut stream = request.into_inner();
        let (tx, rx) = tokio::sync::mpsc::channel(8);

        tokio::spawn(async move {
            let headers = stream.message().await;
            assert!(
                matches!(
                    headers,
                    Ok(Some(ProcessingRequest {
                        request: Some(processing_request::Request::RequestHeaders(_)),
                        ..
                    }))
                ),
                "first ext_proc message should be RequestHeaders"
            );
            loop {
                match stream.message().await {
                    Ok(Some(msg)) => {
                        if let Some(processing_request::Request::RequestBody(b)) = msg.request
                            && b.end_of_stream
                        {
                            break;
                        }
                    },
                    _ => return,
                }
            }

            let header_resp = build_routing_response(&dest);
            drop(tx.send(Ok(header_resp)).await);

            let body_resp = ProcessingResponse {
                response: Some(processing_response::Response::RequestBody(BodyResponse {
                    response: None,
                })),
                ..Default::default()
            };
            drop(tx.send(Ok(body_resp)).await);
        });

        Ok(tonic::Response::new(Box::pin(
            tokio_stream::wrappers::ReceiverStream::new(rx),
        )))
    }
}

fn build_routing_response(destination: &str) -> ProcessingResponse {
    use praxis_ai_llmd_ext_proc::proto::envoy::service::{
        common::v3::{HeaderValue, HeaderValueOption, header_value_option::HeaderAppendAction},
        ext_proc::v3::{CommonResponse, HeaderMutation},
    };
    ProcessingResponse {
        response: Some(processing_response::Response::RequestHeaders(HeadersResponse {
            response: Some(CommonResponse {
                header_mutation: Some(HeaderMutation {
                    set_headers: vec![HeaderValueOption {
                        header: Some(HeaderValue {
                            key: "x-gateway-destination-endpoint".to_owned(),
                            raw_value: destination.as_bytes().to_vec(),
                            ..Default::default()
                        }),
                        append_action: HeaderAppendAction::OverwriteIfExistsOrAdd.into(),
                        ..Default::default()
                    }],
                    ..Default::default()
                }),
                ..Default::default()
            }),
        })),
        ..Default::default()
    }
}

// -----------------------------------------------------------------------------
// Processor Guard
// -----------------------------------------------------------------------------

struct ProcessorGuard {
    addr: SocketAddr,
    stream_count: Arc<AtomicU32>,
    _shutdown: oneshot::Sender<()>,
}

impl ProcessorGuard {
    fn stream_count(&self) -> u32 {
        self.stream_count.load(Ordering::Relaxed)
    }
}

fn start_routing_processor(destination: &str) -> ProcessorGuard {
    let stream_count = Arc::new(AtomicU32::new(0));
    let processor = SimRoutingProcessor {
        destination: destination.to_owned(),
        stream_count: Arc::clone(&stream_count),
    };

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    let listener = rt.block_on(tokio::net::TcpListener::bind("127.0.0.1:0")).unwrap();
    let addr = listener.local_addr().unwrap();
    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();

    std::thread::spawn(move || {
        rt.block_on(async {
            Server::builder()
                .add_service(ExternalProcessorServer::new(processor))
                .serve_with_incoming_shutdown(tokio_stream::wrappers::TcpListenerStream::new(listener), async {
                    drop(shutdown_rx.await);
                })
                .await
                .unwrap();
        });
    });

    praxis_test_utils::wait_for_tcp(&format!("127.0.0.1:{}", addr.port()));

    ProcessorGuard {
        addr,
        stream_count,
        _shutdown: shutdown_tx,
    }
}

// -----------------------------------------------------------------------------
// Proxy Helper
// -----------------------------------------------------------------------------

fn start_sim_proxy(proxy_port: u16, processor_port: u16) -> praxis_test_utils::ProxyGuard {
    let config_yaml = format!(
        r#"
listeners:
  - name: sim-env
    address: "127.0.0.1:{proxy_port}"
    filter_chains: [sim-chain]

filter_chains:
  - name: sim-chain
    filters:
      - filter: ext_proc
        target: "http://127.0.0.1:{processor_port}"
        message_timeout_ms: 5000
        lifecycle_timeout_ms: 10000
        status_on_error: 503
        processing_mode:
          request_body_mode: full_duplex_streamed
          response_header_mode: skip
      - filter: endpoint_selector
        source_header: x-gateway-destination-endpoint
        required: true
        status_on_required_failure: 503
        strip_header: true
"#
    );

    let config = praxis_core::config::Config::from_yaml(&config_yaml).expect("sim env config should parse");
    let registry = praxis_ai::build_full_registry();
    praxis_test_utils::start_proxy_with_registry(&config, &registry)
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[test]
fn simulator_chat_completion_routes_through_praxis() {
    let sim = start_simulator();
    let proc_guard = start_routing_processor(&sim.endpoint());
    let proxy_port = free_port();
    let _proxy = start_sim_proxy(proxy_port, proc_guard.addr.port());

    let body = format!(
        r#"{{"model":"{}","messages":[{{"role":"user","content":"hello"}}],"max_tokens":5}}"#,
        sim.model()
    );
    let proxy_addr = format!("127.0.0.1:{proxy_port}");
    let raw = praxis_test_utils::http_send(&proxy_addr, &json_post("/v1/chat/completions", &body));
    let status = praxis_test_utils::parse_status(&raw);
    let response_body = parse_body(&raw);
    assert_eq!(status, 200, "chat completion should succeed through Praxis");
    assert!(
        !response_body.is_empty(),
        "simulator should return a non-empty response body"
    );

    let json: serde_json::Value = serde_json::from_str(&response_body)
        .unwrap_or_else(|e| panic!("response should be valid JSON: {e}\nbody: {response_body}"));
    assert_eq!(
        json.get("model").and_then(|v| v.as_str()),
        Some(sim.model()),
        "response model should match simulator model"
    );
}

#[test]
fn simulator_spoofed_destination_header_ignored() {
    let sim = start_simulator();
    let proc_guard = start_routing_processor(&sim.endpoint());
    let proxy_port = free_port();
    let _proxy = start_sim_proxy(proxy_port, proc_guard.addr.port());

    let body = format!(
        r#"{{"model":"{}","messages":[{{"role":"user","content":"spoof"}}],"max_tokens":5}}"#,
        sim.model()
    );
    let request = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n\
         Content-Type: application/json\r\n\
         x-gateway-destination-endpoint: 10.99.99.99:9999\r\n\
         Content-Length: {}\r\n\r\n{}",
        body.len(),
        body
    );
    let proxy_addr = format!("127.0.0.1:{proxy_port}");
    let raw = praxis_test_utils::http_send(&proxy_addr, &request);
    let status = praxis_test_utils::parse_status(&raw);
    assert_eq!(
        status, 200,
        "spoofed destination header should be ignored; request should route to simulator"
    );
}

#[test]
fn simulator_repeated_requests_no_crosstalk() {
    let sim = start_simulator();
    let proc_guard = start_routing_processor(&sim.endpoint());
    let proxy_port = free_port();
    let _proxy = start_sim_proxy(proxy_port, proc_guard.addr.port());

    let proxy_addr = format!("127.0.0.1:{proxy_port}");
    let baseline = proc_guard.stream_count();
    for i in 0..3 {
        let body = format!(
            r#"{{"model":"{}","messages":[{{"role":"user","content":"repeat {i}"}}],"max_tokens":5}}"#,
            sim.model()
        );
        let raw = praxis_test_utils::http_send(&proxy_addr, &json_post("/v1/chat/completions", &body));
        let status = praxis_test_utils::parse_status(&raw);
        assert_eq!(status, 200, "request {i} should succeed");
    }
    let streams = proc_guard.stream_count() - baseline;
    assert_eq!(
        streams, 3,
        "each request should use one Process stream (used {streams})"
    );
}

#[test]
fn simulator_health_endpoint_reachable() {
    let sim = start_simulator();
    let raw = praxis_test_utils::http_send(&sim.endpoint(), "GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n");
    let status = praxis_test_utils::parse_status(&raw);
    assert_eq!(status, 200, "simulator health endpoint should return 200");
}

#[test]
fn simulator_metrics_endpoint_reachable() {
    let sim = start_simulator();
    let raw = praxis_test_utils::http_send(&sim.endpoint(), "GET /metrics HTTP/1.1\r\nHost: localhost\r\n\r\n");
    let status = praxis_test_utils::parse_status(&raw);
    let body = parse_body(&raw);

    assert_eq!(status, 200, "simulator metrics endpoint should return 200");
    assert!(!body.trim().is_empty(), "simulator metrics body should be non-empty");
}

#[test]
fn simulator_processor_failure_returns_status_on_error() {
    let sim = start_simulator();
    let unused_port = free_port();
    let proxy_port = free_port();
    let _proxy = start_sim_proxy(proxy_port, unused_port);

    let body = format!(
        r#"{{"model":"{}","messages":[{{"role":"user","content":"fail"}}],"max_tokens":5}}"#,
        sim.model()
    );
    let proxy_addr = format!("127.0.0.1:{proxy_port}");
    let raw = praxis_test_utils::http_send(&proxy_addr, &json_post("/v1/chat/completions", &body));
    let status = praxis_test_utils::parse_status(&raw);
    assert_eq!(status, 503, "processor failure should return status_on_error 503");
}
