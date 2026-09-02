// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Replay input and scripted response resource bounds.

use std::io::Read as _;

use base64::{engine::general_purpose::STANDARD, read::DecoderReader};

use super::{
    FixtureError, RecordedBody, SseFrame,
    schema::{
        DocumentValidationLimits, LIVE_CAPTURE_STRUCTURE_LIMITS, SseParseLimits, is_json_content_type,
        validate_json_bytes_with_limits,
    },
};

/// Maximum scenario request body accepted before any replay networking.
pub(super) const MAX_SCENARIO_REQUEST_BODY_BYTES: usize = 8 * 1024 * 1024;

/// Maximum scripted or captured response body accepted by replay.
pub(super) const MAX_SCRIPTED_RESPONSE_BODY_BYTES: usize = 16 * 1024 * 1024;

/// Maximum number of canonical frames in one SSE body.
pub(super) const MAX_SSE_FRAME_COUNT: usize = 4_096;

/// Maximum canonical wire size of one SSE frame.
pub(super) const MAX_SSE_FRAME_BYTES: usize = 1024 * 1024;

/// Parses a bounded response while enforcing SSE limits before frame allocation.
pub(super) fn parse_response_body(content_type: Option<&str>, bytes: &[u8]) -> Result<RecordedBody, FixtureError> {
    parse_response_body_with_json_limits(content_type, bytes, LIVE_CAPTURE_STRUCTURE_LIMITS)
}

/// Parses one captured request after allocation-light JSON structural preflight.
pub(super) fn parse_request_body(content_type: Option<&str>, bytes: &[u8]) -> Result<RecordedBody, FixtureError> {
    if !bytes.is_empty() && content_type.is_some_and(is_json_content_type) {
        validate_json_bytes_with_limits(bytes, LIVE_CAPTURE_STRUCTURE_LIMITS)?;
    }
    RecordedBody::from_http(content_type, bytes)
}

/// Preflights JSON structure before materialization under explicit limits.
fn parse_response_body_with_json_limits(
    content_type: Option<&str>,
    bytes: &[u8],
    json_limits: DocumentValidationLimits,
) -> Result<RecordedBody, FixtureError> {
    if !bytes.is_empty() && content_type.is_some_and(is_json_content_type) {
        validate_json_bytes_with_limits(bytes, json_limits)?;
    }
    RecordedBody::from_http_with_sse_limits(
        content_type,
        bytes,
        SseParseLimits {
            max_frames: MAX_SSE_FRAME_COUNT,
            max_frame_bytes: MAX_SSE_FRAME_BYTES,
        },
    )
}

/// Validates one scenario body without allocating a second full body buffer.
pub(super) fn validate_request_body(body: &RecordedBody) -> Result<(), FixtureError> {
    validate_request_body_with_limit(body, MAX_SCENARIO_REQUEST_BODY_BYTES)
}

/// Validates one scenario request against an explicit canonical byte ceiling.
pub(super) fn validate_request_body_with_limit(body: &RecordedBody, max_bytes: usize) -> Result<(), FixtureError> {
    validate_body(body, max_bytes, "scenario request body exceeded replay limit")
}

/// Validates one scripted response body without allocating a second full body buffer.
pub(super) fn validate_response_body(body: &RecordedBody) -> Result<(), FixtureError> {
    validate_body(
        body,
        MAX_SCRIPTED_RESPONSE_BODY_BYTES,
        "scripted response body exceeded replay limit",
    )
}

/// Bounded-decodes one recorded request body for sanitization or safety checks.
pub(super) fn decode_request_base64(data: &str) -> Result<Vec<u8>, FixtureError> {
    decode_base64_with_limit(
        data,
        MAX_SCENARIO_REQUEST_BODY_BYTES,
        "scenario request body exceeded replay limit",
    )
}

/// Bounded-decodes one recorded response body for sanitization or safety checks.
pub(super) fn decode_response_base64(data: &str) -> Result<Vec<u8>, FixtureError> {
    decode_base64_with_limit(
        data,
        MAX_SCRIPTED_RESPONSE_BODY_BYTES,
        "scripted response body exceeded replay limit",
    )
}

/// Returns whether a validated recorded body renders at least one wire byte.
pub(super) fn body_has_rendered_content(body: &RecordedBody) -> bool {
    match body {
        RecordedBody::Empty => false,
        RecordedBody::Json { .. } => true,
        RecordedBody::Sse { frames, done } => *done || !frames.is_empty(),
        RecordedBody::Base64 { data } => !data.is_empty(),
    }
}

/// Validates renderability, grammar, and canonical size for one recorded body.
fn validate_body(body: &RecordedBody, max_bytes: usize, size_error: &'static str) -> Result<(), FixtureError> {
    match body {
        RecordedBody::Empty => Ok(()),
        RecordedBody::Json { value } => validate_json_size(value, max_bytes, size_error),
        RecordedBody::Base64 { data } => validate_base64_size(data, max_bytes, size_error),
        RecordedBody::Sse { frames, done } => validate_sse(frames, *done, max_bytes, size_error),
    }
}

/// Counts serialized JSON bytes with a writer that refuses the first excess byte.
fn validate_json_size(
    value: &serde_json::Value,
    max_bytes: usize,
    size_error: &'static str,
) -> Result<(), FixtureError> {
    let mut writer = LimitWriter::new(max_bytes);
    if serde_json::to_writer(&mut writer, value).is_err() {
        return Err(runtime_error(if writer.exceeded {
            size_error
        } else {
            "recorded JSON body could not be rendered"
        }));
    }
    Ok(())
}

/// Streams Base64 decoding through a small stack buffer while counting bytes.
fn validate_base64_size(data: &str, max_bytes: usize, size_error: &'static str) -> Result<(), FixtureError> {
    let mut decoder = DecoderReader::new(data.as_bytes(), &STANDARD);
    let mut decoded = 0_usize;
    let mut buffer = [0_u8; 8 * 1024];
    loop {
        let read = decoder
            .read(&mut buffer)
            .map_err(|_source| runtime_error("recorded Base64 body is invalid"))?;
        if read == 0 {
            return Ok(());
        }
        decoded = decoded.checked_add(read).ok_or_else(|| runtime_error(size_error))?;
        if decoded > max_bytes {
            return Err(runtime_error(size_error));
        }
    }
}

/// Streams Base64 into one fallibly allocated buffer while enforcing its decoded ceiling.
fn decode_base64_with_limit(data: &str, max_bytes: usize, size_error: &'static str) -> Result<Vec<u8>, FixtureError> {
    let estimated = data.len().saturating_mul(3).div_ceil(4).min(max_bytes);
    let mut decoded = Vec::new();
    decoded
        .try_reserve_exact(estimated)
        .map_err(|_source| runtime_error("recorded Base64 body allocation failed"))?;

    let mut decoder = DecoderReader::new(data.as_bytes(), &STANDARD);
    let mut buffer = [0_u8; 8 * 1024];
    loop {
        let read = decoder
            .read(&mut buffer)
            .map_err(|_source| runtime_error("recorded Base64 body is invalid"))?;
        if read == 0 {
            return Ok(decoded);
        }
        let total = decoded
            .len()
            .checked_add(read)
            .ok_or_else(|| runtime_error(size_error))?;
        if total > max_bytes {
            return Err(runtime_error(size_error));
        }
        decoded
            .try_reserve_exact(read)
            .map_err(|_source| runtime_error("recorded Base64 body allocation failed"))?;
        decoded.extend_from_slice(&buffer[..read]);
    }
}

/// Validates SSE grammar and both per-frame and aggregate limits.
fn validate_sse(
    frames: &[SseFrame],
    done: bool,
    max_bytes: usize,
    size_error: &'static str,
) -> Result<(), FixtureError> {
    if frames.len() > MAX_SSE_FRAME_COUNT {
        return Err(runtime_error("recorded SSE frame count exceeded replay limit"));
    }
    let mut total = 0_usize;
    for frame in frames {
        let frame_bytes = canonical_sse_frame_size(frame)?;
        if frame_bytes > MAX_SSE_FRAME_BYTES {
            return Err(runtime_error("recorded SSE frame exceeded replay limit"));
        }
        total = total
            .checked_add(frame_bytes)
            .ok_or_else(|| runtime_error(size_error))?;
        if total > max_bytes {
            return Err(runtime_error(size_error));
        }
    }
    if done {
        total = total
            .checked_add(b"data: [DONE]\n\n".len())
            .ok_or_else(|| runtime_error(size_error))?;
        if total > max_bytes {
            return Err(runtime_error(size_error));
        }
    }
    Ok(())
}

/// Computes canonical wire size while rejecting line-injection grammar.
fn canonical_sse_frame_size(frame: &SseFrame) -> Result<usize, FixtureError> {
    let mut size = 1_usize;
    if let Some(event) = &frame.event {
        validate_single_line_sse_field(event)?;
        size = checked_size_add(size, b"event: \n".len(), event.len())?;
    }
    if frame.data.contains('\r') {
        return Err(runtime_error("recorded SSE data contains a carriage return"));
    }
    for line in frame.data.split('\n') {
        size = checked_size_add(size, b"data: \n".len(), line.len())?;
    }
    if let Some(id) = &frame.id {
        validate_single_line_sse_field(id)?;
        size = checked_size_add(size, b"id: \n".len(), id.len())?;
    }
    if let Some(retry) = frame.retry {
        size = checked_size_add(size, b"retry: \n".len(), retry.to_string().len())?;
    }
    Ok(size)
}

/// Rejects newlines in SSE fields that must occupy one wire line.
fn validate_single_line_sse_field(value: &str) -> Result<(), FixtureError> {
    if value.contains(['\r', '\n']) {
        Err(runtime_error("recorded SSE field contains a newline"))
    } else {
        Ok(())
    }
}

/// Adds one canonical `name: value\n` line size with overflow protection.
fn checked_size_add(current: usize, line_overhead: usize, value: usize) -> Result<usize, FixtureError> {
    current
        .checked_add(line_overhead)
        .and_then(|size| size.checked_add(value))
        .ok_or_else(|| runtime_error("recorded SSE frame size overflowed"))
}

/// A no-allocation JSON writer that errors immediately beyond its limit.
struct LimitWriter {
    /// Bytes accepted so far.
    written: usize,
    /// Inclusive byte ceiling.
    limit: usize,
    /// Whether the writer rejected an excess write.
    exceeded: bool,
}

impl LimitWriter {
    /// Creates an empty counter with an inclusive limit.
    const fn new(limit: usize) -> Self {
        Self {
            written: 0,
            limit,
            exceeded: false,
        }
    }
}

impl std::io::Write for LimitWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let Some(total) = self.written.checked_add(buf.len()) else {
            self.exceeded = true;
            return Err(std::io::Error::other("recorded body size overflow"));
        };
        if total > self.limit {
            self.exceeded = true;
            return Err(std::io::Error::other("recorded body exceeded replay limit"));
        }
        self.written = total;
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// Creates an opaque replay validation error.
fn runtime_error(message: &'static str) -> FixtureError {
    FixtureError::ReplayRuntime { message }
}

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::indexing_slicing, clippy::unwrap_used, reason = "tests")]
mod tests {
    use base64::{Engine as _, engine::general_purpose::STANDARD};
    use serde_json::Value;

    use super::{
        MAX_SCENARIO_REQUEST_BODY_BYTES, MAX_SCRIPTED_RESPONSE_BODY_BYTES, MAX_SSE_FRAME_BYTES, MAX_SSE_FRAME_COUNT,
        parse_response_body, parse_response_body_with_json_limits, validate_request_body, validate_response_body,
    };
    use crate::inference_fixture::{RecordedBody, SseFrame, schema::DocumentValidationLimits};

    #[test]
    fn response_json_structure_is_rejected_before_dense_value_materialization() {
        let limits = DocumentValidationLimits {
            max_nodes: 3,
            max_container_entries: 3,
            max_decoded_string_bytes: 100,
            max_depth: 8,
        };

        parse_response_body_with_json_limits(Some("application/json"), b"[0,0]", limits).unwrap();
        assert!(parse_response_body_with_json_limits(Some("application/json"), b"[0,0,0]", limits).is_err());
    }

    #[test]
    fn request_json_accepts_exact_limit_and_rejects_limit_plus_one() {
        let mut body = RecordedBody::Json {
            value: Value::String("x".repeat(MAX_SCENARIO_REQUEST_BODY_BYTES - 2)),
        };
        validate_request_body(&body).unwrap();

        let RecordedBody::Json {
            value: Value::String(text),
        } = &mut body
        else {
            unreachable!();
        };
        text.push('x');
        assert!(validate_request_body(&body).is_err());
    }

    #[test]
    fn response_json_and_base64_accept_exact_limit_and_reject_limit_plus_one() {
        let mut json_body = RecordedBody::Json {
            value: Value::String("x".repeat(MAX_SCRIPTED_RESPONSE_BODY_BYTES - 2)),
        };
        validate_response_body(&json_body).unwrap();
        let RecordedBody::Json {
            value: Value::String(text),
        } = &mut json_body
        else {
            unreachable!();
        };
        text.push('x');
        assert!(validate_response_body(&json_body).is_err());

        let mut decoded = vec![0_u8; MAX_SCRIPTED_RESPONSE_BODY_BYTES];
        let mut base64_body = RecordedBody::Base64 {
            data: STANDARD.encode(&decoded),
        };
        validate_response_body(&base64_body).unwrap();
        decoded.push(0);
        base64_body = RecordedBody::Base64 {
            data: STANDARD.encode(decoded),
        };
        assert!(validate_response_body(&base64_body).is_err());
    }

    #[test]
    fn sse_accepts_exact_frame_and_count_limits_and_rejects_limit_plus_one() {
        let frame_overhead = b"data: \n\n".len();
        let mut body = RecordedBody::Sse {
            frames: vec![frame("x".repeat(MAX_SSE_FRAME_BYTES - frame_overhead))],
            done: false,
        };
        validate_response_body(&body).unwrap();
        let RecordedBody::Sse { frames, .. } = &mut body else {
            unreachable!();
        };
        frames[0].data.push('x');
        assert!(validate_response_body(&body).is_err());

        let mut frames = std::iter::repeat_with(|| frame(String::new()))
            .take(MAX_SSE_FRAME_COUNT)
            .collect::<Vec<_>>();
        body = RecordedBody::Sse {
            frames: std::mem::take(&mut frames),
            done: false,
        };
        validate_response_body(&body).unwrap();
        let RecordedBody::Sse { frames, .. } = &mut body else {
            unreachable!();
        };
        frames.push(frame(String::new()));
        assert!(validate_response_body(&body).is_err());
    }

    #[test]
    fn response_sse_parser_enforces_exact_frame_and_count_limits_before_materialization() {
        let frame_overhead = b"data: \n\n".len();
        let mut exact_frame = Vec::with_capacity(MAX_SSE_FRAME_BYTES);
        exact_frame.extend_from_slice(b"data: ");
        exact_frame.extend(std::iter::repeat_n(b'x', MAX_SSE_FRAME_BYTES - frame_overhead));
        exact_frame.extend_from_slice(b"\n\n");
        parse_response_body(Some("text/event-stream"), &exact_frame).unwrap();
        exact_frame.insert(exact_frame.len() - 2, b'x');
        assert!(parse_response_body(Some("text/event-stream"), &exact_frame).is_err());

        let one_frame = b"data: \n\n";
        let mut exact_count = one_frame.repeat(MAX_SSE_FRAME_COUNT);
        parse_response_body(Some("text/event-stream"), &exact_count).unwrap();
        exact_count.extend_from_slice(one_frame);
        assert!(parse_response_body(Some("text/event-stream"), &exact_count).is_err());

        assert!(parse_response_body(Some("text/event-stream"), b"data: \xff\n\n").is_err());
    }

    fn frame(data: String) -> SseFrame {
        SseFrame {
            event: None,
            data,
            id: None,
            retry: None,
        }
    }
}
