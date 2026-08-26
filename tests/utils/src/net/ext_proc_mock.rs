// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Shared mock llm-d `ext_proc` routing processor for integration tests.

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
use tokio::sync::oneshot;
use tonic::transport::Server;

use super::wait_for_tcp;

struct MockRoutingProcessor {
    destination: String,
    stream_count: Arc<AtomicU32>,
}

#[async_trait]
impl ExternalProcessor for MockRoutingProcessor {
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

            while let Ok(Some(message)) = stream.message().await {
                if let Some(processing_request::Request::RequestBody(body)) = message.request
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

/// Running mock processor and its invocation counter.
pub struct MockProcessorGuard {
    addr: SocketAddr,
    stream_count: Arc<AtomicU32>,
    _shutdown: oneshot::Sender<()>,
}

impl MockProcessorGuard {
    /// Bound processor address.
    #[must_use]
    pub fn addr(&self) -> SocketAddr {
        self.addr
    }

    /// Bound processor port.
    #[must_use]
    pub fn port(&self) -> u16 {
        self.addr.port()
    }

    /// Number of processing streams accepted by the mock.
    #[must_use]
    pub fn stream_count(&self) -> u32 {
        self.stream_count.load(Ordering::Relaxed)
    }
}

/// Start a mock processor that selects `destination` for every request.
#[must_use]
pub fn start_mock_routing_processor(destination: &str) -> MockProcessorGuard {
    let stream_count = Arc::new(AtomicU32::new(0));
    let processor = MockRoutingProcessor {
        destination: destination.to_owned(),
        stream_count: Arc::clone(&stream_count),
    };
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("mock processor runtime should build");
    let listener = runtime
        .block_on(tokio::net::TcpListener::bind("127.0.0.1:0"))
        .expect("mock processor should bind");
    let addr = listener.local_addr().expect("mock processor should have an address");
    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();

    std::thread::spawn(move || {
        runtime.block_on(async {
            Server::builder()
                .add_service(ExternalProcessorServer::new(processor))
                .serve_with_incoming_shutdown(tokio_stream::wrappers::TcpListenerStream::new(listener), async {
                    drop(shutdown_rx.await);
                })
                .await
                .expect("mock processor should serve");
        });
    });

    wait_for_tcp(&addr.to_string());
    MockProcessorGuard {
        addr,
        stream_count,
        _shutdown: shutdown_tx,
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
