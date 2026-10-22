// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! AWS `EventStream` binary frame decoder for Bedrock streaming responses.
//!
//! The Bedrock `ConverseStream` API returns responses in the AWS Event
//! Stream binary framing format (`application/vnd.amazon.eventstream`).
//! Each frame has the following on-wire layout:
//!
//! ```text
//! ┌──────────────────────────────────────────────┐
//! │ total_byte_length   (u32 BE)  4 bytes         │
//! │ headers_byte_length (u32 BE)  4 bytes         │
//! │ prelude_crc         (u32 BE)  4 bytes  ──┐    │
//! │                                     CRC of bytes 0..8
//! ├──────────────────────────────────────────────┤
//! │ headers             (variable)               │
//! │   each header:                               │
//! │     name_len  (u8)                           │
//! │     name      (UTF-8, name_len bytes)        │
//! │     type      (u8)  — 7 = string             │
//! │     value_len (u16 BE)                       │
//! │     value     (UTF-8, value_len bytes)       │
//! ├──────────────────────────────────────────────┤
//! │ payload             (variable, JSON)         │
//! ├──────────────────────────────────────────────┤
//! │ message_crc         (u32 BE)  4 bytes  ──┐   │
//! │                                     CRC of bytes 0..(total-4)
//! └──────────────────────────────────────────┘
//! ```
//!
//! The minimum valid frame is 16 bytes (12-byte prelude + 4-byte message
//! CRC, with empty headers and empty payload).
//!
//! ## Design
//!
//! [`EventStreamDecoder`] accumulates raw bytes in a [`BytesMut`] buffer
//! and emits complete [`EventStreamMessage`]s on demand.  It is designed
//! for use inside `on_response_body`, where the proxy delivers TCP chunks
//! of unpredictable size — frames may span multiple chunks, or a single
//! chunk may contain several frames.
//!
//! CRC-32 is validated for both the prelude and the full message before
//! the payload is returned; frames that fail either check produce a
//! [`DecodeError`] and the decoder discards them so the stream can
//! continue.
//!
//! ## What Bedrock sends
//!
//! A typical `ConverseStream` response produces these event types in
//! order:
//!
//! | `:event-type`       | `:message-type` | Payload contents                        |
//! |---------------------|-----------------|-----------------------------------------|
//! | `messageStart`      | `event`         | `{"role":"assistant"}`                  |
//! | `contentBlockStart` | `event`         | `{"contentBlockIndex":0,"start":{...}}` |
//! | `contentBlockDelta` | `event`         | `{"contentBlockIndex":0,"delta":{...}}` |
//! | `contentBlockStop`  | `event`         | `{"contentBlockIndex":0}`               |
//! | `messageStop`       | `event`         | `{"stopReason":"end_turn",...}`         |
//! | `metadata`          | `event`         | `{"usage":{...},"metrics":{...}}`       |
//!
//! Exception frames use `:message-type` = `"exception"` and carry an
//! `:exception-type` header (e.g. `"throttlingException"`).

use bytes::{Buf as _, BytesMut};
use thiserror::Error;

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// Byte length of the prelude: `total_length` (4) + `headers_length` (4)
/// + `prelude_crc` (4).
const PRELUDE_LEN: usize = 12;

/// Byte length of the trailing message CRC field.
const MESSAGE_CRC_LEN: usize = 4;

/// Minimum frame size: prelude (12) + message CRC (4).
const MIN_FRAME_LEN: usize = PRELUDE_LEN + MESSAGE_CRC_LEN;

/// AWS `EventStream` header value type code for UTF-8 strings.
const HEADER_TYPE_STRING: u8 = 7;
/// Header value type: raw bytes blob.
const HEADER_TYPE_BYTES: u8 = 6;
/// Header value type: boolean true (no payload).
const HEADER_TYPE_BOOL_TRUE: u8 = 0;
/// Header value type: boolean false (no payload).
const HEADER_TYPE_BOOL_FALSE: u8 = 1;
/// Header value type: 1-byte integer.
const HEADER_TYPE_BYTE: u8 = 2;
/// Header value type: 2-byte integer.
const HEADER_TYPE_SHORT: u8 = 3;
/// Header value type: 4-byte integer.
const HEADER_TYPE_INT: u8 = 4;
/// Header value type: 8-byte integer.
const HEADER_TYPE_LONG: u8 = 5;
/// Header value type: 8-byte timestamp.
const HEADER_TYPE_TIMESTAMP: u8 = 8;
/// Header value type: 16-byte UUID.
const HEADER_TYPE_UUID: u8 = 9;

// -----------------------------------------------------------------------------
// Public Types
// -----------------------------------------------------------------------------

/// A fully decoded AWS `EventStream` frame.
///
/// Only the fields consumed by the Bedrock response translator are
/// retained.  All others are advanced past during parsing but not stored.
#[derive(Debug, Clone)]
pub(crate) struct EventStreamMessage {
    /// Well-known frame headers extracted from the binary header block.
    pub headers: Vec<EventStreamHeader>,
    /// Raw payload bytes.  For Bedrock content events this is a JSON
    /// object; for exception frames it is also JSON.
    pub payload: Bytes,
}

// Re-export so callers can hold `Bytes` without an extra `use`.
use bytes::Bytes;

impl EventStreamMessage {
    /// Return the string value of the named header, or `None` if the
    /// header is absent or is not a string-typed value.
    pub(crate) fn header_str(&self, name: &str) -> Option<&str> {
        self.headers.iter().find_map(|h| {
            if h.name == name {
                match &h.value {
                    HeaderValue::String(s) => Some(s.as_str()),
                    HeaderValue::Other => None,
                }
            } else {
                None
            }
        })
    }

    /// Convenience: return the `:event-type` header value.
    pub(crate) fn event_type(&self) -> Option<&str> {
        self.header_str(":event-type")
    }

    /// Convenience: return the `:message-type` header value.
    pub(crate) fn message_type(&self) -> Option<&str> {
        self.header_str(":message-type")
    }

    /// Return `true` when this frame is an exception
    /// (`:message-type` = `"exception"`).
    pub(crate) fn is_exception(&self) -> bool {
        self.message_type() == Some("exception")
    }
}

/// A single header decoded from an `EventStream` frame.
#[derive(Debug, Clone)]
pub(crate) struct EventStreamHeader {
    /// Header name (e.g. `":event-type"`, `":message-type"`).
    pub name: String,
    /// Decoded value — only `String` is meaningful to Bedrock callers.
    pub value: HeaderValue,
}

/// The value of an `EventStream` header field.
///
/// Only string-typed headers (type `0x07`) carry values that the Bedrock
/// translator reads.  Every other wire type advances the cursor but
/// collapses to [`Self::Other`].
#[non_exhaustive]
#[derive(Debug, Clone)]
pub(crate) enum HeaderValue {
    /// UTF-8 string (wire type `0x07`).
    String(String),
    /// Any non-string type whose value is irrelevant to this caller.
    Other,
}

// -----------------------------------------------------------------------------
// Decoder
// -----------------------------------------------------------------------------

/// Incremental decoder for the AWS `EventStream` binary framing protocol.
///
/// Call [`push`] to supply raw bytes received from the upstream, then
/// call [`decode`] (or [`decode_all`]) to drain complete frames.  The
/// decoder owns an internal [`BytesMut`] ring-buffer so partial frames
/// are preserved across calls.
///
/// [`push`]: EventStreamDecoder::push
/// [`decode`]: EventStreamDecoder::decode
/// [`decode_all`]: EventStreamDecoder::decode_all
#[derive(Debug)]
pub(crate) struct EventStreamDecoder {
    /// Accumulation buffer for incoming raw bytes.
    buf: BytesMut,
    /// Maximum accepted frame size, including framing overhead.
    max_frame_len: usize,
}

impl EventStreamDecoder {
    /// Create a new decoder with a default 8-KiB initial buffer capacity.
    #[cfg(test)]
    pub(crate) fn new() -> Self {
        Self::with_max_frame_len(4 * 1_048_576)
    }

    /// Create a decoder that rejects frames larger than `max_frame_len`.
    pub(crate) fn with_max_frame_len(max_frame_len: usize) -> Self {
        Self {
            buf: BytesMut::with_capacity(8 * 1024),
            max_frame_len,
        }
    }

    /// Return whether no partial frame bytes remain buffered.
    pub(crate) fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    /// Append raw bytes to the internal buffer.
    ///
    /// This is cheap — `BytesMut` never copies if capacity is available.
    pub(crate) fn push(&mut self, data: &[u8]) {
        self.buf.extend_from_slice(data);
    }

    /// Attempt to decode the next complete frame from the buffer.
    ///
    /// Returns:
    /// - `Ok(Some(msg))` — a complete, CRC-validated frame was decoded and consumed from the buffer.
    /// - `Ok(None)` — not enough bytes yet; call [`push`] with more data and retry.
    /// - `Err(e)` — the frame is structurally invalid (bad length, CRC mismatch, or malformed headers).  The offending
    ///   bytes are consumed so the decoder can attempt to continue.
    ///
    /// [`push`]: EventStreamDecoder::push
    #[expect(
        clippy::too_many_lines,
        reason = "sequential parse phases; splitting would scatter the CRC/length checks across multiple functions"
    )]
    #[expect(
        clippy::indexing_slicing,
        reason = "all accesses guarded: buf.len() >= PRELUDE_LEN ensures indices 0–11; split_to ensures frame indices are valid"
    )]
    pub(crate) fn decode(&mut self) -> Result<Option<EventStreamMessage>, DecodeError> {
        // Need at least the prelude to read lengths.
        if self.buf.len() < PRELUDE_LEN {
            return Ok(None);
        }

        // Read the two length fields without advancing — we may not have
        // the full frame yet.
        let total_len = u32::from_be_bytes([self.buf[0], self.buf[1], self.buf[2], self.buf[3]]) as usize;
        let headers_len = u32::from_be_bytes([self.buf[4], self.buf[5], self.buf[6], self.buf[7]]) as usize;

        // A frame must be at least MIN_FRAME_LEN bytes.
        if total_len < MIN_FRAME_LEN {
            // Discard this malformed prelude and report an error.
            self.buf.advance(PRELUDE_LEN);
            return Err(DecodeError::FrameTooShort(total_len));
        }

        if total_len > self.max_frame_len {
            self.buf.clear();
            return Err(DecodeError::FrameTooLarge {
                total_len,
                max_frame_len: self.max_frame_len,
            });
        }

        // Wait for the full frame to arrive.
        if self.buf.len() < total_len {
            return Ok(None);
        }

        // Split exactly `total_len` bytes from the front — this is O(1)
        // thanks to `BytesMut`'s reference-counted backing store.
        let frame = self.buf.split_to(total_len).freeze();

        // --- Validate prelude CRC (covers bytes 0..8) ---
        let prelude_crc_stored = u32::from_be_bytes([frame[8], frame[9], frame[10], frame[11]]);
        let prelude_crc_computed = crc32fast::hash(&frame[..8]);
        if prelude_crc_computed != prelude_crc_stored {
            return Err(DecodeError::PreludeCrcMismatch {
                stored: prelude_crc_stored,
                computed: prelude_crc_computed,
            });
        }

        // --- Validate message CRC (covers bytes 0..(total_len - 4)) ---
        let msg_crc_stored = u32::from_be_bytes([
            frame[total_len - 4],
            frame[total_len - 3],
            frame[total_len - 2],
            frame[total_len - 1],
        ]);
        let msg_crc_computed = crc32fast::hash(&frame[..total_len - MESSAGE_CRC_LEN]);
        if msg_crc_computed != msg_crc_stored {
            return Err(DecodeError::MessageCrcMismatch {
                stored: msg_crc_stored,
                computed: msg_crc_computed,
            });
        }

        // --- Parse headers ---
        let headers_start = PRELUDE_LEN;
        let headers_end = headers_start + headers_len;

        // Guard: headers must not overlap the trailing CRC.
        if headers_end > total_len - MESSAGE_CRC_LEN {
            return Err(DecodeError::HeadersExceedFrame { headers_len, total_len });
        }
        let headers = decode_headers(&frame[headers_start..headers_end])?;

        // --- Payload sits between headers and the trailing CRC ---
        let payload_start = headers_end;
        let payload_end = total_len - MESSAGE_CRC_LEN;
        let payload = frame.slice(payload_start..payload_end);

        Ok(Some(EventStreamMessage { headers, payload }))
    }

    /// Drain all complete frames currently available in the buffer.
    ///
    /// Stops at the first `Err` and returns it, leaving subsequent (valid)
    /// frames in the buffer.  Returns an empty `Vec` if no frames are
    /// complete yet.
    pub(crate) fn decode_all(&mut self) -> Result<Vec<EventStreamMessage>, DecodeError> {
        let mut messages = Vec::new();
        while let Some(msg) = self.decode()? {
            messages.push(msg);
        }
        Ok(messages)
    }
}

// -----------------------------------------------------------------------------
// Header Parsing
// -----------------------------------------------------------------------------

/// Parse the binary header block into a `Vec<EventStreamHeader>`.
///
/// The AWS `EventStream` header encoding is:
///
/// ```text
/// [name_len: u8][name: UTF-8][type: u8][value...]
/// ```
///
/// Value encoding depends on `type`:
/// - `0x07` (String): `[value_len: u16 BE][value: UTF-8]`
/// - `0x06` (Bytes):  `[value_len: u16 BE][value: bytes]` — advanced, not kept
/// - `0x00` / `0x01` (Bool): no payload
/// - `0x02` (Byte): 1-byte payload
/// - `0x03` (Short): 2-byte payload
/// - `0x04` (Int): 4-byte payload
/// - `0x05` / `0x08` (Long / Timestamp): 8-byte payload
/// - `0x09` (UUID): 16-byte payload
#[expect(
    clippy::too_many_lines,
    reason = "exhaustively handles every AWS EventStream header value type; splitting by type would not improve clarity"
)]
#[expect(
    clippy::indexing_slicing,
    reason = "all slice accesses are preceded by a length check that returns TruncatedHeader"
)]
fn decode_headers(mut data: &[u8]) -> Result<Vec<EventStreamHeader>, DecodeError> {
    let mut headers = Vec::new();

    while !data.is_empty() {
        // --- Name ---
        let name_len = *data.first().ok_or(DecodeError::TruncatedHeader)? as usize;
        data = data.get(1..).ok_or(DecodeError::TruncatedHeader)?;

        if data.len() < name_len {
            return Err(DecodeError::TruncatedHeader);
        }
        let name = std::str::from_utf8(&data[..name_len])
            .map_err(|_e| DecodeError::InvalidUtf8InHeader)?
            .to_owned();
        data = &data[name_len..];

        // --- Type ---
        let value_type = *data.first().ok_or(DecodeError::TruncatedHeader)?;
        data = data.get(1..).ok_or(DecodeError::TruncatedHeader)?;

        // --- Value ---
        let value = match value_type {
            HEADER_TYPE_STRING => {
                let (s, rest) = read_length_prefixed_string(data)?;
                data = rest;
                HeaderValue::String(s)
            },
            HEADER_TYPE_BYTES => {
                let (len, rest) = read_u16_prefix_len(data)?;
                if rest.len() < len {
                    return Err(DecodeError::TruncatedHeader);
                }
                data = rest.get(len..).ok_or(DecodeError::TruncatedHeader)?;
                HeaderValue::Other
            },
            HEADER_TYPE_BOOL_TRUE | HEADER_TYPE_BOOL_FALSE => HeaderValue::Other,
            HEADER_TYPE_BYTE => {
                data = skip(data, 1)?;
                HeaderValue::Other
            },
            HEADER_TYPE_SHORT => {
                data = skip(data, 2)?;
                HeaderValue::Other
            },
            HEADER_TYPE_INT => {
                data = skip(data, 4)?;
                HeaderValue::Other
            },
            HEADER_TYPE_LONG | HEADER_TYPE_TIMESTAMP => {
                data = skip(data, 8)?;
                HeaderValue::Other
            },
            HEADER_TYPE_UUID => {
                data = skip(data, 16)?;
                HeaderValue::Other
            },
            other => return Err(DecodeError::UnknownHeaderType(other)),
        };

        headers.push(EventStreamHeader { name, value });
    }

    Ok(headers)
}

// -----------------------------------------------------------------------------
// Internal Helpers
// -----------------------------------------------------------------------------

/// Read a `u16 BE` length prefix followed by that many UTF-8 bytes.
/// Returns `(string, remaining_slice)`.
#[expect(clippy::indexing_slicing, reason = "length checked above before the slice")]
fn read_length_prefixed_string(data: &[u8]) -> Result<(String, &[u8]), DecodeError> {
    let (len, rest) = read_u16_prefix_len(data)?;
    if rest.len() < len {
        return Err(DecodeError::TruncatedHeader);
    }
    let s = std::str::from_utf8(&rest[..len])
        .map_err(|_e| DecodeError::InvalidUtf8InHeader)?
        .to_owned();
    Ok((s, &rest[len..]))
}

/// Read a `u16 BE` from the front of `data` and return `(value, rest)`.
#[expect(
    clippy::indexing_slicing,
    reason = "bounds checked: data.len() >= 2 is asserted above"
)]
fn read_u16_prefix_len(data: &[u8]) -> Result<(usize, &[u8]), DecodeError> {
    if data.len() < 2 {
        return Err(DecodeError::TruncatedHeader);
    }
    let len = u16::from_be_bytes([data[0], data[1]]) as usize;
    Ok((len, &data[2..]))
}

/// Advance `data` by `n` bytes, returning the remainder.
#[expect(
    clippy::indexing_slicing,
    reason = "bounds checked: data.len() >= n is asserted above"
)]
fn skip(data: &[u8], n: usize) -> Result<&[u8], DecodeError> {
    if data.len() < n {
        return Err(DecodeError::TruncatedHeader);
    }
    Ok(&data[n..])
}

// -----------------------------------------------------------------------------
// Error Type
// -----------------------------------------------------------------------------

/// Errors that can occur while decoding an AWS `EventStream` frame.
#[derive(Debug, Error)]
#[non_exhaustive]
pub(crate) enum DecodeError {
    /// The `total_byte_length` field is smaller than the minimum valid frame.
    #[error("frame total_length {0} is below minimum ({MIN_FRAME_LEN})")]
    FrameTooShort(usize),

    /// The declared frame length exceeds the configured allocation limit.
    #[error("frame total_length {total_len} exceeds configured maximum {max_frame_len}")]
    FrameTooLarge {
        /// Declared frame length.
        total_len: usize,
        /// Configured maximum frame length.
        max_frame_len: usize,
    },

    /// The prelude CRC-32 does not match.
    #[error("prelude CRC-32 mismatch: stored {stored:#010x}, computed {computed:#010x}")]
    PreludeCrcMismatch {
        /// CRC-32 read from the wire.
        stored: u32,
        /// CRC-32 recomputed locally.
        computed: u32,
    },

    /// The message CRC-32 does not match.
    #[error("message CRC-32 mismatch: stored {stored:#010x}, computed {computed:#010x}")]
    MessageCrcMismatch {
        /// CRC-32 read from the wire.
        stored: u32,
        /// CRC-32 recomputed locally.
        computed: u32,
    },

    /// The declared `headers_byte_length` exceeds the frame boundaries.
    #[error("headers_length {headers_len} overflows frame total_length {total_len}")]
    HeadersExceedFrame {
        /// Declared length of the headers section.
        headers_len: usize,
        /// Declared total frame length.
        total_len: usize,
    },

    /// The header block ended unexpectedly mid-field.
    #[error("header block truncated")]
    TruncatedHeader,

    /// A header name or string value contained invalid UTF-8.
    #[error("invalid UTF-8 in header")]
    InvalidUtf8InHeader,

    /// Encountered an unrecognised header value type byte.
    #[error("unknown header value type: {0:#04x}")]
    UnknownHeaderType(u8),
}

// -----------------------------------------------------------------------------
// Test helpers (also used from integration tests in response.rs)
// -----------------------------------------------------------------------------

/// Build a complete, CRC-correct AWS `EventStream` binary frame from
/// string headers and a raw payload.
///
/// This is used exclusively in tests; production code only decodes.
#[cfg(test)]
#[expect(clippy::expect_used, reason = "test helper: inputs are compile-time constants")]
#[expect(
    clippy::indexing_slicing,
    reason = "frame slice is populated by extend_from_slice above"
)]
pub(crate) fn build_frame(headers: &[(&str, &str)], payload: &[u8]) -> Vec<u8> {
    // Encode headers.
    let mut header_bytes: Vec<u8> = Vec::new();
    for (name, value) in headers {
        let name_b = name.as_bytes();
        // name_len (u8) + name
        header_bytes.push(u8::try_from(name_b.len()).expect("header name too long for u8 length prefix"));
        header_bytes.extend_from_slice(name_b);
        // type = 7 (string)
        header_bytes.push(HEADER_TYPE_STRING);
        // value_len (u16 BE) + value
        let val_b = value.as_bytes();
        header_bytes.extend_from_slice(
            &u16::try_from(val_b.len())
                .expect("header value too long for u16 length prefix")
                .to_be_bytes(),
        );
        header_bytes.extend_from_slice(val_b);
    }

    // total_length = 12 (prelude) + headers + payload + 4 (message CRC)
    let total_len = u32::try_from(MIN_FRAME_LEN + header_bytes.len() + payload.len())
        .expect("frame too large for u32 total_length");
    let headers_len = u32::try_from(header_bytes.len()).expect("headers too large for u32");

    let mut frame: Vec<u8> = Vec::with_capacity(total_len as usize);
    frame.extend_from_slice(&total_len.to_be_bytes());
    frame.extend_from_slice(&headers_len.to_be_bytes());

    // Prelude CRC over the first 8 bytes.
    frame.extend_from_slice(&crc32fast::hash(&frame[..8]).to_be_bytes());

    frame.extend_from_slice(&header_bytes);
    frame.extend_from_slice(payload);

    // Message CRC over everything so far.
    frame.extend_from_slice(&crc32fast::hash(&frame).to_be_bytes());

    frame
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
#[expect(clippy::unwrap_used, clippy::indexing_slicing, clippy::expect_used, reason = "tests")]
mod tests {
    use super::*;

    // ── Helpers ────────────────────────────────────────────────────────────

    fn event_frame(event_type: &str, payload: &[u8]) -> Vec<u8> {
        build_frame(&[(":message-type", "event"), (":event-type", event_type)], payload)
    }

    fn exception_frame(exception_type: &str, payload: &[u8]) -> Vec<u8> {
        build_frame(
            &[(":message-type", "exception"), (":exception-type", exception_type)],
            payload,
        )
    }

    // ── build_frame sanity ─────────────────────────────────────────────────

    #[test]
    fn build_frame_produces_decodable_message() {
        let payload = br#"{"role":"assistant"}"#;
        let raw = event_frame("messageStart", payload);

        let mut dec = EventStreamDecoder::new();
        dec.push(&raw);
        let msg = dec.decode().unwrap().expect("must yield a message");

        assert_eq!(msg.event_type(), Some("messageStart"));
        assert_eq!(msg.message_type(), Some("event"));
        assert_eq!(&*msg.payload, payload);
    }

    // ── Single frame ───────────────────────────────────────────────────────

    #[test]
    fn decode_content_block_delta() {
        let payload = br#"{"contentBlockIndex":0,"delta":{"text":"Hello"}}"#;
        let raw = event_frame("contentBlockDelta", payload);

        let mut dec = EventStreamDecoder::new();
        dec.push(&raw);
        let msg = dec.decode().unwrap().unwrap();

        assert_eq!(msg.event_type(), Some("contentBlockDelta"));
        assert_eq!(&*msg.payload, payload);
    }

    #[test]
    fn decode_message_stop() {
        let payload = br#"{"stopReason":"end_turn","additionalModelResponseFields":null}"#;
        let raw = event_frame("messageStop", payload);

        let mut dec = EventStreamDecoder::new();
        dec.push(&raw);
        let msg = dec.decode().unwrap().unwrap();

        assert_eq!(msg.event_type(), Some("messageStop"));
    }

    #[test]
    fn declared_frame_larger_than_limit_is_rejected_before_buffering_payload() {
        let mut dec = EventStreamDecoder::with_max_frame_len(64);
        let total_len = 65_u32;
        let mut prelude = Vec::from(total_len.to_be_bytes());
        prelude.extend_from_slice(&0_u32.to_be_bytes());
        prelude.extend_from_slice(&crc32fast::hash(&prelude).to_be_bytes());
        dec.push(&prelude);

        assert!(matches!(
            dec.decode(),
            Err(DecodeError::FrameTooLarge {
                total_len: 65,
                max_frame_len: 64
            })
        ));
        assert!(dec.is_empty());
    }

    #[test]
    fn decode_empty_payload() {
        let raw = event_frame("contentBlockStop", b"");

        let mut dec = EventStreamDecoder::new();
        dec.push(&raw);
        let msg = dec.decode().unwrap().unwrap();

        assert_eq!(msg.event_type(), Some("contentBlockStop"));
        assert!(msg.payload.is_empty());
    }

    #[test]
    fn decode_exception_frame() {
        let payload = br#"{"message":"throttled"}"#;
        let raw = exception_frame("throttlingException", payload);

        let mut dec = EventStreamDecoder::new();
        dec.push(&raw);
        let msg = dec.decode().unwrap().unwrap();

        assert!(msg.is_exception());
        assert_eq!(msg.header_str(":exception-type"), Some("throttlingException"));
        assert_eq!(&*msg.payload, payload);
    }

    // ── Multiple frames ────────────────────────────────────────────────────

    #[test]
    fn decode_all_returns_frames_in_order() {
        let f1 = event_frame("messageStart", br#"{"role":"assistant"}"#);
        let f2 = event_frame("contentBlockDelta", br#"{"contentBlockIndex":0,"delta":{"text":"Hi"}}"#);
        let f3 = event_frame("messageStop", br#"{"stopReason":"end_turn"}"#);

        let mut all = Vec::new();
        all.extend_from_slice(&f1);
        all.extend_from_slice(&f2);
        all.extend_from_slice(&f3);

        let mut dec = EventStreamDecoder::new();
        dec.push(&all);
        let msgs = dec.decode_all().unwrap();

        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[0].event_type(), Some("messageStart"));
        assert_eq!(msgs[1].event_type(), Some("contentBlockDelta"));
        assert_eq!(msgs[2].event_type(), Some("messageStop"));
    }

    #[test]
    fn decode_all_concatenated_single_push() {
        // Two frames pushed as one contiguous byte buffer.
        let mut buf = event_frame("messageStart", b"{}");
        buf.extend_from_slice(&event_frame("contentBlockDelta", b"{}"));

        let mut dec = EventStreamDecoder::new();
        dec.push(&buf);
        let msgs = dec.decode_all().unwrap();
        assert_eq!(msgs.len(), 2);
    }

    // ── Partial / split chunks ─────────────────────────────────────────────

    #[test]
    fn decode_partial_frame_returns_none_until_complete() {
        let raw = event_frame("messageStart", br#"{"role":"assistant"}"#);

        let mut dec = EventStreamDecoder::new();

        // Feed one byte at a time until the last byte.
        for (i, byte) in raw[..raw.len() - 1].iter().enumerate() {
            dec.push(std::slice::from_ref(byte));
            let result = dec.decode().unwrap();
            assert!(result.is_none(), "byte {i}: expected None, got Some");
        }

        // Feed the last byte — now the frame is complete.
        dec.push(&raw[raw.len() - 1..]);
        let msg = dec.decode().unwrap().expect("must decode after final byte");
        assert_eq!(msg.event_type(), Some("messageStart"));
    }

    #[test]
    fn decode_frame_split_at_midpoint() {
        let raw = event_frame("contentBlockDelta", br#"{"delta":{"text":"x"}}"#);
        let mid = raw.len() / 2;

        let mut dec = EventStreamDecoder::new();
        dec.push(&raw[..mid]);
        assert!(dec.decode().unwrap().is_none(), "must need more bytes");

        dec.push(&raw[mid..]);
        let msg = dec.decode().unwrap().expect("must decode after second half");
        assert_eq!(msg.event_type(), Some("contentBlockDelta"));
    }

    #[test]
    fn second_frame_follows_first_partial() {
        // Push two frames; the second arrives split across two chunks.
        let f1 = event_frame("messageStart", b"{}");
        let f2 = event_frame("messageStop", br#"{"stopReason":"end_turn"}"#);

        let mut dec = EventStreamDecoder::new();
        dec.push(&f1);

        // Deliver first frame completely.
        let m1 = dec.decode().unwrap().expect("first frame");
        assert_eq!(m1.event_type(), Some("messageStart"));

        // Deliver second frame split in two.
        let mid = f2.len() / 2;
        dec.push(&f2[..mid]);
        assert!(dec.decode().unwrap().is_none());

        dec.push(&f2[mid..]);
        let m2 = dec.decode().unwrap().expect("second frame");
        assert_eq!(m2.event_type(), Some("messageStop"));
    }

    // ── Error cases ────────────────────────────────────────────────────────

    #[test]
    fn decode_frame_too_short_returns_error() {
        // total_length = 5, which is below MIN_FRAME_LEN (16).
        let mut bad = vec![0_u8; PRELUDE_LEN];
        bad[0..4].copy_from_slice(&5_u32.to_be_bytes());
        // Fill headers_length with 0.
        // Prelude CRC must be correct for us to reach the length check.
        let crc = crc32fast::hash(&bad[..8]);
        bad[8..12].copy_from_slice(&crc.to_be_bytes());

        let mut dec = EventStreamDecoder::new();
        dec.push(&bad);
        assert!(
            matches!(dec.decode(), Err(DecodeError::FrameTooShort(_))),
            "must report FrameTooShort"
        );
    }

    #[test]
    fn decode_bad_prelude_crc() {
        let mut raw = event_frame("messageStart", b"{}");
        // Corrupt the prelude CRC (bytes 8..12).
        raw[8] ^= 0xFF;

        let mut dec = EventStreamDecoder::new();
        dec.push(&raw);
        assert!(
            matches!(dec.decode(), Err(DecodeError::PreludeCrcMismatch { .. })),
            "must report PreludeCrcMismatch"
        );
    }

    #[test]
    fn decode_bad_message_crc() {
        let mut raw = event_frame("messageStart", b"{}");
        // Corrupt the last byte (message CRC).
        let last = raw.len() - 1;
        raw[last] ^= 0xFF;

        let mut dec = EventStreamDecoder::new();
        dec.push(&raw);
        assert!(
            matches!(dec.decode(), Err(DecodeError::MessageCrcMismatch { .. })),
            "must report MessageCrcMismatch"
        );
    }

    #[test]
    fn insufficient_bytes_returns_none() {
        let mut dec = EventStreamDecoder::new();
        dec.push(&[0x00, 0x00, 0x00]); // only 3 bytes, prelude is 12
        assert!(dec.decode().unwrap().is_none());
    }

    // ── Payload integrity ─────────────────────────────────────────────────

    #[test]
    fn payload_bytes_match_exactly() {
        // Use a regular string then convert to bytes to avoid non-ASCII
        // raw-byte-string literal restrictions.
        let text = "{\"contentBlockIndex\":0,\"delta\":{\"text\":\"hello world\"}}";
        let json = text.as_bytes();
        let raw = event_frame("contentBlockDelta", json);

        let mut dec = EventStreamDecoder::new();
        dec.push(&raw);
        let msg = dec.decode().unwrap().unwrap();

        assert_eq!(&*msg.payload, json);
    }

    #[test]
    fn metadata_frame_decoded() {
        let payload = br#"{"usage":{"inputTokens":10,"outputTokens":5,"totalTokens":15}}"#;
        let raw = event_frame("metadata", payload);

        let mut dec = EventStreamDecoder::new();
        dec.push(&raw);
        let msg = dec.decode().unwrap().unwrap();

        assert_eq!(msg.event_type(), Some("metadata"));
        assert_eq!(&*msg.payload, payload);
    }

    // ── Header helpers ────────────────────────────────────────────────────

    #[test]
    fn header_str_returns_none_for_absent_header() {
        let raw = event_frame("messageStart", b"{}");
        let mut dec = EventStreamDecoder::new();
        dec.push(&raw);
        let msg = dec.decode().unwrap().unwrap();
        assert!(msg.header_str(":no-such-header").is_none());
    }

    #[test]
    fn is_exception_false_for_event_frame() {
        let raw = event_frame("contentBlockDelta", b"{}");
        let mut dec = EventStreamDecoder::new();
        dec.push(&raw);
        let msg = dec.decode().unwrap().unwrap();
        assert!(!msg.is_exception());
    }

    #[test]
    fn is_exception_true_for_exception_frame() {
        let raw = exception_frame("validationException", b"{}");
        let mut dec = EventStreamDecoder::new();
        dec.push(&raw);
        let msg = dec.decode().unwrap().unwrap();
        assert!(msg.is_exception());
    }
}
