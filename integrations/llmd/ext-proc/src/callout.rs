// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! gRPC stream management for the `ext_proc` filter.
//!
//! Opens a bidirectional `Process` stream to the external processor,
//! sends a single [`ProcessingRequest`], and receives a single
//! [`ProcessingResponse`] within a configurable timeout.
//!
//! [`ProcessingRequest`]: crate::proto::envoy::service::ext_proc::v3::ProcessingRequest
//! [`ProcessingResponse`]: crate::proto::envoy::service::ext_proc::v3::ProcessingResponse

use std::time::Duration;

use futures::stream;
use praxis_filter::{FilterAction, FilterError, HttpFilterContext};
use tonic::transport::Channel;

use crate::{
    Phase,
    mutations::{apply_headers_response, immediate_to_rejection, request_to_proto_headers, response_to_proto_headers},
    proto::envoy::service::ext_proc::v3::{
        ProcessingRequest, ProcessingResponse, external_processor_client::ExternalProcessorClient, processing_request,
        processing_response,
    },
};

// -----------------------------------------------------------------------------
// CalloutError
// -----------------------------------------------------------------------------

/// Errors that can occur during a gRPC callout.
#[derive(Debug, thiserror::Error)]
pub(crate) enum CalloutError {
    /// gRPC transport or protocol error.
    #[error("ext_proc gRPC error: {0}")]
    Grpc(#[from] tonic::Status),

    /// The per-message timeout expired.
    #[error("ext_proc message timeout")]
    Timeout,

    /// The server closed the stream without sending a response.
    #[error("ext_proc server closed stream without response")]
    EmptyStream,
}

// -----------------------------------------------------------------------------
// Public callout functions
// -----------------------------------------------------------------------------

/// Send request headers to the external processor and apply mutations.
///
/// Opens a `Process` stream, sends a `RequestHeaders` message, and
/// waits for one response within `timeout`. Returns [`FilterAction`]
/// indicating whether the pipeline should continue or reject.
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "kept for focused request-header callout tests")
)]
pub(crate) async fn process_request_headers(
    channel: Channel,
    target: &str,
    timeout: Duration,
    max_timeout: Option<Duration>,
    ctx: &mut HttpFilterContext<'_>,
) -> Result<FilterAction, FilterError> {
    let headers = request_to_proto_headers(ctx);
    let request = ProcessingRequest {
        request: Some(processing_request::Request::RequestHeaders(headers)),
        ..Default::default()
    };

    let response = send_and_receive(channel, request, timeout, max_timeout, target).await?;
    dispatch_response(&response, ctx, Phase::Request)
}

/// Send response headers to the external processor and apply mutations.
///
/// Same pattern as [`process_request_headers`] but wraps
/// `ResponseHeaders` and operates during the response phase.
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "kept for focused response-header callout tests")
)]
pub(crate) async fn process_response_headers(
    channel: Channel,
    target: &str,
    timeout: Duration,
    max_timeout: Option<Duration>,
    ctx: &mut HttpFilterContext<'_>,
) -> Result<FilterAction, FilterError> {
    let headers = response_to_proto_headers(ctx);
    let request = ProcessingRequest {
        request: Some(processing_request::Request::ResponseHeaders(headers)),
        ..Default::default()
    };

    let response = send_and_receive(channel, request, timeout, max_timeout, target).await?;
    dispatch_response(&response, ctx, Phase::Response)
}

// -----------------------------------------------------------------------------
// Private helpers
// -----------------------------------------------------------------------------

/// Open a `Process` stream, send one request, and receive one response.
///
/// Each callout opens its own stream. The initial timeout covers
/// stream setup and the first message. If the processor responds
/// with `override_message_timeout` (and no `response` oneof), a
/// new deadline replaces the original for the subsequent read,
/// clamped to `max_timeout`.
async fn send_and_receive(
    channel: Channel,
    request: ProcessingRequest,
    timeout: Duration,
    max_timeout: Option<Duration>,
    target: &str,
) -> Result<ProcessingResponse, FilterError> {
    let deadline = tokio::time::Instant::now()
        .checked_add(timeout)
        .ok_or(CalloutError::Timeout)?;

    let mut streaming = match open_stream(channel, request, deadline).await {
        Ok(s) => s,
        Err(CalloutError::Grpc(e)) => {
            tracing::warn!(target = %target, error = %e, "ext_proc stream open failed");
            return Err(CalloutError::Grpc(e).into());
        },
        Err(CalloutError::Timeout) => {
            tracing::warn!(target = %target, "ext_proc callout timed out during stream open");
            return Err(CalloutError::Timeout.into());
        },
        Err(CalloutError::EmptyStream) => return Err(CalloutError::EmptyStream.into()),
    };

    let result = receive_with_override(&mut streaming, deadline, max_timeout, target).await;

    match result {
        Ok(response) => Ok(response),
        Err(e) => {
            tracing::warn!(target = %target, error = %e, "ext_proc callout failed");
            Err(e.into())
        },
    }
}

/// Open the processor stream before the initial callout deadline.
async fn open_stream(
    channel: Channel,
    request: ProcessingRequest,
    deadline: tokio::time::Instant,
) -> Result<tonic::Streaming<ProcessingResponse>, CalloutError> {
    tokio::time::timeout_at(deadline, async {
        let mut client = ExternalProcessorClient::new(channel);
        let request_stream = stream::once(async { request });
        let rpc = client.process(request_stream).await.map_err(CalloutError::Grpc)?;
        Ok::<_, CalloutError>(rpc.into_inner())
    })
    .await
    .map_err(|_elapsed| CalloutError::Timeout)?
}

/// Read the first response, handling `override_message_timeout`.
///
/// The first read uses the original `deadline`. If the processor
/// sends `override_message_timeout` with no `response` oneof, a
/// new absolute deadline is computed from the current time and the
/// override duration (clamped to `max_timeout`), replacing the
/// original. Without a configured `max_timeout`, overrides are
/// ignored and the response is returned as-is.
async fn receive_with_override(
    streaming: &mut tonic::Streaming<ProcessingResponse>,
    deadline: tokio::time::Instant,
    max_timeout: Option<Duration>,
    target: &str,
) -> Result<ProcessingResponse, CalloutError> {
    let resp = tokio::time::timeout_at(deadline, next_message(streaming))
        .await
        .map_err(|_elapsed| CalloutError::Timeout)??;

    if resp.response.is_some() {
        return Ok(resp);
    }

    let Some(override_dur) = parse_timeout_override(&resp, max_timeout) else {
        return Ok(resp);
    };

    tracing::debug!(
        target = %target,
        override_ms = override_dur.as_millis(),
        "ext_proc: processor requested timeout override"
    );

    let new_deadline = tokio::time::Instant::now()
        .checked_add(override_dur)
        .ok_or(CalloutError::Timeout)?;
    tokio::time::timeout_at(new_deadline, next_message(streaming))
        .await
        .map_err(|_elapsed| CalloutError::Timeout)?
}

/// Read the next message from the stream.
async fn next_message(
    streaming: &mut tonic::Streaming<ProcessingResponse>,
) -> Result<ProcessingResponse, CalloutError> {
    streaming
        .message()
        .await
        .map_err(CalloutError::Grpc)?
        .ok_or(CalloutError::EmptyStream)
}

/// Extract and clamp the `override_message_timeout` from a response.
///
/// Returns `None` if the field is absent, the duration is zero, or
/// `max_timeout` is not configured (overrides require an upper bound).
pub(crate) fn parse_timeout_override(resp: &ProcessingResponse, max_timeout: Option<Duration>) -> Option<Duration> {
    let max = max_timeout?;
    let proto_dur = resp.override_message_timeout.as_ref()?;
    let dur = parse_override_duration(proto_dur)?;

    let clamped = dur.min(max);
    if clamped < dur {
        tracing::warn!(
            requested_ms = dur.as_millis(),
            clamped_ms = clamped.as_millis(),
            max_ms = max.as_millis(),
            "ext_proc: override_message_timeout clamped to max_message_timeout"
        );
    }

    Some(clamped)
}

/// Parse an Envoy protobuf duration using the protobuf range and sign rules.
fn parse_override_duration(value: &prost_types::Duration) -> Option<Duration> {
    if value.seconds < 0 || value.seconds > 315_576_000_000 {
        return None;
    }
    if value.nanos < 0 || value.nanos >= 1_000_000_000 {
        return None;
    }
    #[expect(clippy::cast_sign_loss, reason = "negative values rejected above")]
    let duration = Duration::new(value.seconds as u64, value.nanos as u32);
    (!duration.is_zero()).then_some(duration)
}

/// Route a [`ProcessingResponse`] variant to the correct mutation handler.
///
/// Returns [`FilterAction::Continue`] for header mutations or
/// [`FilterAction::Reject`] for immediate responses. Unexpected
/// response types produce a [`FilterError`].
fn dispatch_response(
    response: &ProcessingResponse,
    ctx: &mut HttpFilterContext<'_>,
    phase: Phase,
) -> Result<FilterAction, FilterError> {
    let Some(resp) = &response.response else {
        return Ok(FilterAction::Continue);
    };

    match (resp, phase) {
        (processing_response::Response::RequestHeaders(hr), Phase::Request)
        | (processing_response::Response::ResponseHeaders(hr), Phase::Response) => {
            apply_headers_response(hr, ctx, phase);
            Ok(FilterAction::Continue)
        },
        (processing_response::Response::ImmediateResponse(imm), _) => Ok(immediate_to_rejection(imm)),
        (other, _) => {
            let variant = response_variant_name(other);
            Err(format!("ext_proc: unexpected response type '{variant}' during {phase} phase").into())
        },
    }
}

/// Returns a human-readable name for a [`processing_response::Response`] variant.
fn response_variant_name(resp: &processing_response::Response) -> &'static str {
    match resp {
        processing_response::Response::RequestHeaders(_) => "RequestHeaders",
        processing_response::Response::ResponseHeaders(_) => "ResponseHeaders",
        processing_response::Response::RequestBody(_) => "RequestBody",
        processing_response::Response::ResponseBody(_) => "ResponseBody",
        processing_response::Response::RequestTrailers(_) => "RequestTrailers",
        processing_response::Response::ResponseTrailers(_) => "ResponseTrailers",
        processing_response::Response::ImmediateResponse(_) => "ImmediateResponse",
    }
}
