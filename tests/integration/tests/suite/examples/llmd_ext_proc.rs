// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Tests for the llm-d ext_proc routing example configuration.

use std::{
    collections::HashMap,
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
use praxis_test_utils::{free_port, http_post, start_backend_with_shutdown, start_proxy};
use tokio::sync::oneshot;
use tonic::transport::Server;

// -----------------------------------------------------------------------------
// Mock Processor
// -----------------------------------------------------------------------------

struct RoutingProcessor {
    destination: String,
    stream_count: Arc<AtomicU32>,
}

#[async_trait]
impl ExternalProcessor for RoutingProcessor {
    type ProcessStream = Pin<Box<dyn futures::Stream<Item = Result<ProcessingResponse, tonic::Status>> + Send>>;

    async fn process(
        &self,
        request: tonic::Request<tonic::Streaming<ProcessingRequest>>,
    ) -> Result<tonic::Response<Self::ProcessStream>, tonic::Status> {
        self.stream_count.fetch_add(1, Ordering::Relaxed);
        let destination = self.destination.clone();
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

            while let Ok(Some(msg)) = stream.message().await {
                if let Some(processing_request::Request::RequestBody(body)) = msg.request
                    && body.end_of_stream
                {
                    break;
                }
            }

            drop(tx.send(Ok(build_routing_response(&destination))).await);
            drop(tx.send(Ok(build_body_continue_response())).await);
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

fn build_body_continue_response() -> ProcessingResponse {
    ProcessingResponse {
        response: Some(processing_response::Response::RequestBody(BodyResponse {
            response: None,
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
    let processor = RoutingProcessor {
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
// Tests
// -----------------------------------------------------------------------------

#[test]
fn llmd_ext_proc_routing_example_routes_to_processor_selected_endpoint() {
    let backend = start_backend_with_shutdown("llmd-selected-backend");
    let processor = start_routing_processor(&format!("127.0.0.1:{}", backend.port()));
    let proxy_port = free_port();

    let config = super::load_example_config(
        "llmd-ext-proc-routing.yaml",
        proxy_port,
        HashMap::from([("127.0.0.1:3001", processor.addr.port())]),
    );
    let proxy = start_proxy(&config);

    let (status, body) = http_post(
        proxy.addr(),
        "/v1/chat/completions",
        r#"{"model":"llmd-demo","messages":[{"role":"user","content":"hello"}]}"#,
    );

    assert_eq!(status, 200, "example request should return 200");
    assert_eq!(
        body, "llmd-selected-backend",
        "request should route to the processor-selected backend"
    );
    assert!(
        processor.stream_count() >= 1,
        "example should invoke the ext_proc processor"
    );
}
