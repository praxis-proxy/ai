// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Bounded SSE wire-frame preservation for Vertex response translation.

/// Incrementally retains original frame bytes, including non-data fields.
pub(super) struct WireParser {
    /// Number of emitted wire boundaries; only the first can carry a stream BOM.
    pub completed_frames: usize,
    /// A failed stream emits one error and suppresses later success events.
    pub failed: bool,
    /// In-flight frame; ownership moves to the caller at each boundary.
    frame: Vec<u8>,
    /// Maximum retained bytes for one frame.
    limit: usize,
    /// Number of non-terminator bytes in the current line.
    line_bytes: usize,
    /// A CR may still acquire a following LF in the next chunk.
    pending_cr: bool,
}

impl WireParser {
    /// Construct a parser with a per-frame memory limit.
    pub fn new(limit: usize) -> Self {
        Self {
            completed_frames: 0,
            failed: false,
            frame: Vec::new(),
            limit,
            line_bytes: 0,
            pending_cr: false,
        }
    }

    /// Return complete frames, failing on overflow or an incomplete final frame.
    pub fn parse_chunk(&mut self, chunk: &[u8], end_of_stream: bool) -> Result<Vec<Vec<u8>>, ParseFailure> {
        let mut frames = Vec::new();
        for &byte in chunk {
            if let Err(message) = self.push_byte(byte, &mut frames) {
                return Err(self.failure(frames, message));
            }
        }
        if end_of_stream {
            if self.pending_cr {
                self.finish_line(&mut frames);
                self.pending_cr = false;
            }
            if !self.frame.is_empty() {
                return Err(self.failure(frames, "Vertex AI ended an incomplete SSE frame"));
            }
        }
        Ok(frames)
    }

    /// Release the failed partial frame while retaining completed events.
    fn failure(&mut self, frames: Vec<Vec<u8>>, message: &'static str) -> ParseFailure {
        self.failed = true;
        self.frame = Vec::new();
        ParseFailure { frames, message }
    }

    /// Move the completed frame out when a blank line terminates it.
    fn finish_line(&mut self, frames: &mut Vec<Vec<u8>>) {
        if self.line_bytes == 0 {
            frames.push(std::mem::take(&mut self.frame));
            self.completed_frames = self.completed_frames.saturating_add(1);
        }
        self.line_bytes = 0;
    }

    /// Recognize LF, CRLF, and bare CR without normalizing the wire bytes.
    fn push_byte(&mut self, byte: u8, frames: &mut Vec<Vec<u8>>) -> Result<(), &'static str> {
        if self.pending_cr {
            self.pending_cr = false;
            if byte == b'\n' {
                self.retain_byte(byte)?;
                self.finish_line(frames);
                return Ok(());
            }
            self.finish_line(frames);
        }
        self.retain_byte(byte)?;
        match byte {
            b'\r' => self.pending_cr = true,
            b'\n' => self.finish_line(frames),
            _ => self.line_bytes = self.line_bytes.saturating_add(1),
        }
        Ok(())
    }

    /// Enforce the frame limit for every wire byte, including CRLF suffixes.
    fn retain_byte(&mut self, byte: u8) -> Result<(), &'static str> {
        if self.frame.len() >= self.limit {
            return Err("Vertex AI exceeded the SSE frame size limit");
        }
        self.frame.push(byte);
        Ok(())
    }
}

/// A failed tail and the preceding complete frames from the same chunk.
#[derive(Debug)]
pub(super) struct ParseFailure {
    /// Valid events that must reach the client before the error.
    pub frames: Vec<Vec<u8>>,
    /// Client-safe explanation without upstream response data.
    pub message: &'static str,
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "tests")]
mod tests {
    use super::WireParser;

    #[test]
    fn wire_frames_preserve_line_endings_across_every_split() {
        for source in [
            b": ping\r\nid: 3\r\ndata: hi\r\n\r\n".as_slice(),
            b"data: hi\r\r",
            b"data: hi\n\n",
            b"event: content_block_delta\ndata: {\ndata: \"type\":\"content_block_delta\",\ndata: \"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}\ndata: }\n\n",
        ] {
            for split in 0..=source.len() {
                let (first, second) = source.split_at(split);
                let mut parser = WireParser::new(256);
                let mut frames = parser.parse_chunk(first, false).unwrap();
                frames.extend(parser.parse_chunk(second, true).unwrap());
                assert_eq!(frames.concat(), source, "wire bytes must survive split {split}");
            }
        }
    }

    #[test]
    fn incomplete_or_oversized_frames_fail_explicitly() {
        assert!(WireParser::new(8).parse_chunk(b"data: too large\n\n", false).is_err());
        assert!(WireParser::new(256).parse_chunk(b"data: truncated\n", true).is_err());
    }

    #[test]
    fn failure_keeps_completed_frames_before_oversized_or_incomplete_tail() {
        let valid = b"data: valid\n\n";
        for (limit, tail) in [
            (32, b"data: this tail exceeds the frame limit\n\n".as_slice()),
            (256, b"data: incomplete"),
        ] {
            let mut parser = WireParser::new(limit);
            let error = parser
                .parse_chunk(&[valid.as_slice(), tail].concat(), true)
                .unwrap_err();
            assert_eq!(error.frames, vec![valid.to_vec()]);
            assert!(parser.failed);
            assert!(!error.message.is_empty());
        }
    }
}
