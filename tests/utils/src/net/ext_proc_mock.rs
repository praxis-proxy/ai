// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Shared mock llm-d `ext_proc` routing processor for integration tests.

use std::{
    net::SocketAddr,
    pin::Pin,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU32, Ordering},
    },
};

use async_trait::async_trait;
use praxis_ai_llmd_ext_proc::proto::envoy::service::{
    common::v3::{HeaderValue, HeaderValueOption, header_value_option::HeaderAppendAction},
    ext_proc::v3::{
        BodyMutation, BodyResponse, CommonResponse, HeaderMutation, HeadersResponse, HttpHeaders, ProcessingRequest,
        ProcessingResponse, StreamedBodyResponse, body_mutation,
        external_processor_server::{ExternalProcessor, ExternalProcessorServer},
        processing_request, processing_response,
    },
};
use tokio::sync::oneshot;
use tonic::transport::Server;

use super::wait_for_tcp;

/// Observations shared between the gRPC mock and its test owner.
#[derive(Default)]
struct ProcessorCapture {
    /// Number of streams accepted, including streams without headers.
    stream_count: AtomicU32,
    /// First headers message received on each stream that supplied one.
    request_headers: Mutex<Vec<HttpHeaders>>,
    /// Bytes received through request-body EOS, in wire order.
    request_bodies: Mutex<Vec<Vec<u8>>>,
}

/// Full-duplex test processor with an optional configured endpoint.
struct MockRoutingProcessor {
    /// Reply to send after receiving the complete request.
    routing_response: ProcessingResponse,
    /// Reply that completes request-body processing.
    body_response: ProcessingResponse,
    /// Request observations reported to the test.
    capture: Arc<ProcessorCapture>,
}

#[async_trait]
impl ExternalProcessor for MockRoutingProcessor {
    type ProcessStream = Pin<Box<dyn futures::Stream<Item = Result<ProcessingResponse, tonic::Status>> + Send>>;

    #[expect(clippy::too_many_lines, reason = "capture opening headers and body before replying")]
    async fn process(
        &self,
        request: tonic::Request<tonic::Streaming<ProcessingRequest>>,
    ) -> Result<tonic::Response<Self::ProcessStream>, tonic::Status> {
        self.capture.stream_count.fetch_add(1, Ordering::Relaxed);
        let routing_response = self.routing_response.clone();
        let body_response = self.body_response.clone();
        let capture = Arc::clone(&self.capture);
        let mut stream = request.into_inner();
        let (tx, rx) = tokio::sync::mpsc::channel(8);

        tokio::spawn(async move {
            let first_request = stream
                .message()
                .await
                .ok()
                .flatten()
                .and_then(|message| message.request);
            let Some(processing_request::Request::RequestHeaders(headers)) = first_request else {
                panic!("first ext_proc message should be RequestHeaders");
            };
            capture
                .request_headers
                .lock()
                .expect("request headers lock poisoned")
                .push(headers);

            let Some(body) = read_request_body(&mut stream).await else {
                return;
            };
            capture
                .request_bodies
                .lock()
                .expect("request bodies lock poisoned")
                .push(body);
            if tx.send(Ok(routing_response)).await.is_ok() {
                drop(tx.send(Ok(body_response)).await);
            }
        });

        Ok(tonic::Response::new(Box::pin(
            tokio_stream::wrappers::ReceiverStream::new(rx),
        )))
    }
}

/// Capture a complete body without silently accepting a stream that omits EOS.
async fn read_request_body(stream: &mut tonic::Streaming<ProcessingRequest>) -> Option<Vec<u8>> {
    let mut captured = Vec::new();
    while let Ok(Some(message)) = stream.message().await {
        if let Some(processing_request::Request::RequestBody(body)) = message.request {
            captured.extend_from_slice(&body.body);
            if body.end_of_stream {
                return Some(captured);
            }
        }
    }
    None
}

/// Running mock processor, invocation counter, and captured request headers.
pub struct MockProcessorGuard {
    /// Loopback address of the gRPC listener.
    addr: SocketAddr,
    /// Request observations from the server.
    capture: Arc<ProcessorCapture>,
    /// Dropping the sender shuts down the listener.
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
        self.capture.stream_count.load(Ordering::Relaxed)
    }

    /// Take the first `RequestHeaders` message received on each stream.
    ///
    /// # Panics
    /// Panics if the capture lock is poisoned.
    pub fn take_request_headers(&self) -> Vec<HttpHeaders> {
        std::mem::take(
            &mut *self
                .capture
                .request_headers
                .lock()
                .expect("request headers lock poisoned"),
        )
    }

    /// Take the complete bodies received by the processor.
    ///
    /// # Panics
    /// Panics if the capture lock is poisoned.
    pub fn take_request_bodies(&self) -> Vec<Vec<u8>> {
        std::mem::take(
            &mut *self
                .capture
                .request_bodies
                .lock()
                .expect("request bodies lock poisoned"),
        )
    }
}

/// Start a mock processor that selects `destination` for every request.
///
/// # Panics
/// Panics if the mock cannot start its listener or runtime.
#[must_use]
pub fn start_mock_routing_processor(destination: &str) -> MockProcessorGuard {
    start_mock_processor(
        build_routing_response(Some(destination), None),
        build_body_response(None),
    )
}

/// Start a routing mock whose reply also appends one request header.
///
/// # Panics
/// Panics if the mock cannot start its listener or runtime.
#[must_use]
pub fn start_mock_routing_processor_with_reply_header(destination: &str, header: (&str, &str)) -> MockProcessorGuard {
    start_mock_processor(
        build_routing_response(Some(destination), Some(header)),
        build_body_response(None),
    )
}

/// Start a routing mock that also returns replacement request bytes.
///
/// # Panics
/// Panics if the mock cannot start its listener or runtime.
#[must_use]
pub fn start_mock_routing_processor_with_body_replacement(destination: &str, body: &[u8]) -> MockProcessorGuard {
    start_mock_processor(
        build_routing_response(Some(destination), None),
        build_body_response(Some(body)),
    )
}

/// Start a processor that completes the exchange without choosing a worker.
///
/// # Panics
/// Panics if the mock cannot start its listener or runtime.
#[must_use]
pub fn start_mock_processor_without_destination() -> MockProcessorGuard {
    start_mock_processor(build_routing_response(None, None), build_body_response(None))
}

/// Bind the listener before starting its runtime thread.
#[expect(
    clippy::too_many_lines,
    reason = "bind the listener before starting its runtime thread"
)]
fn start_mock_processor(routing_response: ProcessingResponse, body_response: ProcessingResponse) -> MockProcessorGuard {
    let capture = Arc::new(ProcessorCapture::default());
    let processor = MockRoutingProcessor {
        routing_response,
        body_response,
        capture: Arc::clone(&capture),
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
        capture,
        _shutdown: shutdown_tx,
    }
}

/// Build an EPP reply with the configured request-header mutations.
fn build_routing_response(destination: Option<&str>, appended_header: Option<(&str, &str)>) -> ProcessingResponse {
    let destination = destination.map(|value| {
        reply_header(
            "x-gateway-destination-endpoint",
            value,
            HeaderAppendAction::OverwriteIfExistsOrAdd,
        )
    });
    let appended_header =
        appended_header.map(|(key, value)| reply_header(key, value, HeaderAppendAction::AppendIfExistsOrAdd));

    ProcessingResponse {
        response: Some(processing_response::Response::RequestHeaders(HeadersResponse {
            response: Some(CommonResponse {
                header_mutation: Some(HeaderMutation {
                    set_headers: destination.into_iter().chain(appended_header).collect(),
                    ..Default::default()
                }),
                ..Default::default()
            }),
        })),
        ..Default::default()
    }
}

/// Encode one header edit with its append action.
fn reply_header(key: &str, value: &str, action: HeaderAppendAction) -> HeaderValueOption {
    HeaderValueOption {
        header: Some(HeaderValue {
            key: key.to_owned(),
            raw_value: value.as_bytes().to_vec(),
            ..Default::default()
        }),
        append_action: action.into(),
        ..Default::default()
    }
}

/// Complete request-body processing, optionally replacing the body.
fn build_body_response(replacement: Option<&[u8]>) -> ProcessingResponse {
    let response = replacement.map(|body| CommonResponse {
        body_mutation: Some(BodyMutation {
            mutation: Some(body_mutation::Mutation::StreamedResponse(StreamedBodyResponse {
                body: body.to_vec(),
                end_of_stream: true,
            })),
        }),
        ..Default::default()
    });
    ProcessingResponse {
        response: Some(processing_response::Response::RequestBody(BodyResponse { response })),
        ..Default::default()
    }
}
