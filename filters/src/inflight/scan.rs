// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Incremental, resumable scan for a JSON request body's top-level `model`
//! and token-cap fields.
//!
//! [`FieldScan`] is fed body chunks one at a time and carries only small,
//! bounded state across calls: JSON nesting depth,
//! string/escape state, which field (if any) is currently being read, and at
//! most a few hundred bytes for whatever key or value is incomplete. It never
//! buffers the request body itself, so a `model` or token cap that arrives
//! after megabytes of preceding content (e.g. a long `messages` array) is
//! still found, and a value split across a chunk boundary (a key, a string
//! value, or a run of digits) resolves once the next chunk completes it
//! rather than committing a truncated value.

/// Longest top-level key considered as a candidate field name. Longer than
/// any recognized field (`max_completion_tokens` is the longest, at 21
/// bytes), so a legitimate key is never truncated before comparison.
const KEY_CAP: usize = 64;

/// Longest `model` value captured. Comfortably larger than any real model
/// identifier; a longer value is left uncaptured rather than grown without
/// bound.
const MODEL_VALUE_CAP: usize = 512;

/// Longest digit run considered for a token cap. `u64::MAX` has 20 digits;
/// a longer run cannot parse as `u64` regardless of further digits.
const NUMBER_CAP: usize = 20;

/// Top-level field a scan attributes a request to.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Field {
    /// The `model` key.
    Model,
    /// A token-cap key (`max_tokens`, `max_completion_tokens`, or
    /// `max_output_tokens`).
    Cap,
}

/// What a `"..."` string currently being scanned is used for.
#[derive(Clone, Copy, PartialEq, Eq)]
enum StringRole {
    /// A candidate top-level key; compared against known field names on close.
    Key,
    /// The value of a matched `model` key.
    Model,
    /// Any other string (nested content, or a value for a field never
    /// captured as a string); its content is never buffered.
    Other,
}

/// State while inside a `"..."` string literal: which role it plays, its
/// accumulated content, and escape/overflow bookkeeping.
struct StringScan {
    /// What this string's content is used for.
    role: StringRole,
    /// Content accumulated so far, bounded by `role`'s cap.
    buf: String,
    /// Whether the previous character was an unconsumed `\`.
    escaped: bool,
    /// Set once `buf` would exceed its role's cap; the role can no longer
    /// match or be captured, so further characters are dropped.
    overflowed: bool,
}

impl StringScan {
    /// Start scanning a new string literal in the given role.
    fn new(role: StringRole) -> Self {
        Self {
            role,
            buf: String::new(),
            escaped: false,
            overflowed: false,
        }
    }
}

/// Where the scan currently is within the JSON document.
#[derive(Default)]
enum ScanState {
    /// Outside any string, not mid-key or mid-value.
    #[default]
    Idle,
    /// Inside a `"..."` string.
    InString(StringScan),
    /// A candidate top-level key just closed; awaiting `:`.
    AfterKey {
        /// The field this key resolved to, if any.
        field: Option<Field>,
    },
    /// After `:`, awaiting the value's first non-whitespace byte.
    BeforeValue {
        /// The field this value belongs to, if any.
        field: Option<Field>,
    },
    /// Reading a bare scalar (number / `true` / `false` / `null`) up to its
    /// delimiter. `capture` is true only while the digits read so far could
    /// still be a valid `u64` cap value.
    InScalar {
        /// Digits accumulated so far, while `capture` holds.
        buf: String,
        /// Whether `buf` could still become a valid `u64` cap value.
        capture: bool,
    },
}

/// Resumable scan of a JSON request body's top-level `model` and token cap.
///
/// Feed body chunks in order via [`feed_bytes`]. The scan stops consuming
/// further bytes once both fields resolve or the top-level object closes.
///
/// [`feed_bytes`]: Self::feed_bytes
#[derive(Default)]
pub(super) struct FieldScan {
    /// Current object/array nesting depth; the document object is depth 1.
    depth: u32,
    /// Where the scan currently is within the document.
    state: ScanState,
    /// Up to 3 bytes of a UTF-8 sequence split across a chunk boundary.
    partial_utf8: Vec<u8>,
    /// Resolved top-level `model`, once found.
    model: Option<String>,
    /// Resolved token cap, once found.
    cap: Option<u64>,
    /// Set once both fields resolve or the top-level object closes.
    done: bool,
}

impl FieldScan {
    /// The top-level `model` value, once resolved.
    pub(super) fn model(&self) -> Option<&str> {
        self.model.as_deref()
    }

    /// The resolved token cap (`max_tokens` / `max_completion_tokens` /
    /// `max_output_tokens`, whichever appears first), once resolved.
    pub(super) fn cap(&self) -> Option<u64> {
        self.cap
    }

    /// Feed the next body chunk in stream order. A no-op once both fields are
    /// known or the top-level object has closed.
    pub(super) fn feed_bytes(&mut self, bytes: &[u8]) {
        if self.done {
            return;
        }
        if self.partial_utf8.is_empty() {
            self.feed_valid_prefix(bytes);
        } else {
            let mut combined = std::mem::take(&mut self.partial_utf8);
            combined.extend_from_slice(bytes);
            self.feed_valid_prefix(&combined);
        }
    }

    /// Feed as much of `bytes` as is valid UTF-8, skipping any genuinely
    /// invalid sequences and stashing a truncated trailing multi-byte
    /// sequence in `partial_utf8` for the next chunk.
    fn feed_valid_prefix(&mut self, mut bytes: &[u8]) {
        while !bytes.is_empty() && !self.done {
            match std::str::from_utf8(bytes) {
                Ok(text) => {
                    self.feed(text);
                    return;
                },
                Err(err) => {
                    let valid_up_to = err.valid_up_to();
                    if let Some(valid) = bytes.get(..valid_up_to).and_then(|v| std::str::from_utf8(v).ok()) {
                        self.feed(valid);
                    }
                    if let Some(bad_len) = err.error_len() {
                        bytes = bytes.get(valid_up_to.saturating_add(bad_len)..).unwrap_or_default();
                    } else {
                        self.partial_utf8 = bytes.get(valid_up_to..).unwrap_or_default().to_vec();
                        return;
                    }
                },
            }
        }
    }

    /// Feed a complete, valid UTF-8 fragment through the state machine.
    fn feed(&mut self, text: &str) {
        for ch in text.chars() {
            if self.done {
                return;
            }
            self.step(ch);
        }
    }

    /// Advance the state machine by one character.
    fn step(&mut self, ch: char) {
        let state = std::mem::take(&mut self.state);
        self.state = match state {
            ScanState::Idle => self.step_idle(ch),
            ScanState::InString(s) => self.step_in_string(ch, s),
            ScanState::AfterKey { field } => self.step_after_key(ch, field),
            ScanState::BeforeValue { field } => self.step_before_value(ch, field),
            ScanState::InScalar { buf, capture } => self.step_in_scalar(ch, buf, capture),
        };
    }

    /// Advance one character while outside any string, mid-key, or mid-value.
    fn step_idle(&mut self, ch: char) -> ScanState {
        match ch {
            '{' | '[' => {
                self.depth = self.depth.saturating_add(1);
                ScanState::Idle
            },
            '}' | ']' => {
                self.depth = self.depth.saturating_sub(1);
                if self.depth == 0 {
                    self.done = true;
                }
                ScanState::Idle
            },
            '"' => {
                let role = if self.depth == 1 {
                    StringRole::Key
                } else {
                    StringRole::Other
                };
                ScanState::InString(StringScan::new(role))
            },
            _ => ScanState::Idle,
        }
    }

    /// Advance one character through a `"..."` string literal.
    fn step_in_string(&mut self, ch: char, mut s: StringScan) -> ScanState {
        if s.escaped {
            Self::push_bounded(&mut s, ch);
            s.escaped = false;
            return ScanState::InString(s);
        }
        match ch {
            '\\' => {
                Self::push_bounded(&mut s, ch);
                s.escaped = true;
                ScanState::InString(s)
            },
            '"' => self.close_string(s),
            _ => {
                Self::push_bounded(&mut s, ch);
                ScanState::InString(s)
            },
        }
    }

    /// Handle a string literal's closing quote: resolve a candidate key to a
    /// field, or record a captured `model` value.
    fn close_string(&mut self, s: StringScan) -> ScanState {
        match s.role {
            StringRole::Key => {
                let candidate = if s.overflowed {
                    None
                } else {
                    Self::field_for_key(&s.buf)
                };
                let field = match candidate {
                    Some(Field::Model) if self.model.is_none() => Some(Field::Model),
                    Some(Field::Cap) if self.cap.is_none() => Some(Field::Cap),
                    _ => None,
                };
                ScanState::AfterKey { field }
            },
            StringRole::Model => {
                if !s.overflowed && self.model.is_none() {
                    self.model = Some(s.buf);
                    self.maybe_mark_done();
                }
                ScanState::Idle
            },
            StringRole::Other => ScanState::Idle,
        }
    }

    /// Advance one character while a candidate top-level key awaits its `:`.
    fn step_after_key(&mut self, ch: char, field: Option<Field>) -> ScanState {
        if ch == ':' {
            return ScanState::BeforeValue { field };
        }
        if ch.is_whitespace() {
            return ScanState::AfterKey { field };
        }
        // Malformed input (a key not followed by `:`); resynchronize rather
        // than wedge the scan in a state it can never leave.
        self.step_idle(ch)
    }

    /// Advance one character while awaiting a value's first non-whitespace byte.
    fn step_before_value(&mut self, ch: char, field: Option<Field>) -> ScanState {
        if ch.is_whitespace() {
            return ScanState::BeforeValue { field };
        }
        match ch {
            '"' => Self::start_value_string(field),
            '{' | '[' => {
                self.depth = self.depth.saturating_add(1);
                ScanState::Idle
            },
            ',' | '}' | ']' => self.step_idle(ch), // malformed empty value; resynchronize
            digit if digit.is_ascii_digit() && field == Some(Field::Cap) => ScanState::InScalar {
                buf: String::from(digit),
                capture: true,
            },
            // `true`, `false`, `null`, a negative number, or a digit for a
            // field we don't capture as a number: never ours, just skipped.
            _ => ScanState::InScalar {
                buf: String::new(),
                capture: false,
            },
        }
    }

    /// Start scanning a value's opening quote as the matched field's string
    /// (only `model` captures a string value) or an uncaptured string.
    fn start_value_string(field: Option<Field>) -> ScanState {
        let role = if field == Some(Field::Model) {
            StringRole::Model
        } else {
            StringRole::Other
        };
        ScanState::InString(StringScan::new(role))
    }

    /// Advance one character through a bare scalar value.
    fn step_in_scalar(&mut self, ch: char, mut buf: String, mut capture: bool) -> ScanState {
        if ch == ',' || ch == '}' || ch == ']' || ch.is_whitespace() {
            if capture
                && let Ok(n) = buf.parse::<u64>()
                && self.cap.is_none()
            {
                self.cap = Some(n);
                self.maybe_mark_done();
            }
            return self.step_idle(ch);
        }
        if capture {
            if ch.is_ascii_digit() && buf.len() < NUMBER_CAP {
                buf.push(ch);
            } else {
                // A non-digit (float, exponent, ...) or a run too long to be
                // a valid u64: never a plain integer cap value.
                capture = false;
                buf.clear();
            }
        }
        ScanState::InScalar { buf, capture }
    }

    /// Mark the scan done once both `model` and the cap have resolved.
    fn maybe_mark_done(&mut self) {
        if self.model.is_some() && self.cap.is_some() {
            self.done = true;
        }
    }

    /// Map a top-level key name to the field it feeds, if any.
    fn field_for_key(key: &str) -> Option<Field> {
        match key {
            "model" => Some(Field::Model),
            "max_tokens" | "max_completion_tokens" | "max_output_tokens" => Some(Field::Cap),
            _ => None,
        }
    }

    /// Append `ch` to `s`'s buffer, bounded by its role's cap. Once a buffer
    /// would exceed its bound, mark it overflowed and drop the partial
    /// content; an overflowed buffer can never match or be captured, so there
    /// is nothing left to grow.
    fn push_bounded(s: &mut StringScan, ch: char) {
        if s.overflowed {
            return;
        }
        let cap = match s.role {
            StringRole::Key => KEY_CAP,
            StringRole::Model => MODEL_VALUE_CAP,
            StringRole::Other => return,
        };
        if s.buf.len() + ch.len_utf8() > cap {
            s.overflowed = true;
            s.buf.clear();
            return;
        }
        s.buf.push(ch);
    }
}

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing, reason = "tests")]
mod tests {
    use super::*;

    /// Feed `chunks` in order and return the resolved `(model, cap)`.
    fn scan(chunks: &[&[u8]]) -> (Option<String>, Option<u64>) {
        let mut scan = FieldScan::default();
        for chunk in chunks {
            scan.feed_bytes(chunk);
        }
        (scan.model().map(str::to_owned), scan.cap())
    }

    #[test]
    fn reads_model_and_cap_in_one_chunk() {
        let (model, cap) = scan(&[br#"{"model":"gpt-4","max_tokens":512,"messages":[]}"#]);
        assert_eq!(model.as_deref(), Some("gpt-4"));
        assert_eq!(cap, Some(512));
    }

    #[test]
    fn reads_responses_max_output_tokens() {
        let (model, cap) = scan(&[br#"{"model":"o3","max_output_tokens":2048}"#]);
        assert_eq!(model.as_deref(), Some("o3"));
        assert_eq!(cap, Some(2048));
    }

    #[test]
    fn reads_chat_completions_max_completion_tokens() {
        // Chat Completions deprecated `max_tokens` in favor of
        // `max_completion_tokens`; o-series models only accept the new field.
        let (model, cap) = scan(&[br#"{"model":"o4-mini","max_completion_tokens":256}"#]);
        assert_eq!(model.as_deref(), Some("o4-mini"));
        assert_eq!(cap, Some(256));
    }

    #[test]
    fn missing_fields_are_none() {
        let (model, cap) = scan(&[br#"{"messages":[{"role":"user","content":"hi"}]}"#]);
        assert_eq!(model, None);
        assert_eq!(cap, None);
    }

    #[test]
    fn nested_decoy_never_wins_over_top_level() {
        let (model, cap) =
            scan(&[br#"{"messages":[{"role":"user","model":"decoy","max_tokens":1}],"model":"gpt-4","max_tokens":9}"#]);
        assert_eq!(model.as_deref(), Some("gpt-4"));
        assert_eq!(cap, Some(9));
    }

    #[test]
    fn key_split_across_chunk_boundary() {
        let (model, cap) = scan(&[br#"{"mod"#, br#"el":"gpt-4","max_tokens":5}"#]);
        assert_eq!(model.as_deref(), Some("gpt-4"));
        assert_eq!(cap, Some(5));
    }

    #[test]
    fn model_value_split_across_chunk_boundary() {
        let (model, cap) = scan(&[br#"{"model":"gpt-"#, br#"4","max_tokens":5}"#]);
        assert_eq!(model.as_deref(), Some("gpt-4"));
        assert_eq!(cap, Some(5));
    }

    #[test]
    fn cap_digit_run_split_across_chunk_boundary() {
        // A cap cut off mid-number must not resolve to the truncated prefix:
        // 1|000 must become 1000, not 1.
        let (model, cap) = scan(&[br#"{"model":"gpt-4","max_tokens":1"#, b"000}"]);
        assert_eq!(model.as_deref(), Some("gpt-4"));
        assert_eq!(cap, Some(1000));
    }

    #[test]
    fn nested_depth_correct_across_chunk_boundary() {
        // The second chunk starts inside the nested `metadata` object: a scan
        // that lost nesting depth across the boundary would read
        // {"model":"decoy"} as top-level.
        let (model, cap) = scan(&[
            br#"{"metadata":"#,
            br#"{"model":"decoy"},"model":"gpt-4","max_tokens":5}"#,
        ]);
        assert_eq!(model.as_deref(), Some("gpt-4"));
        assert_eq!(cap, Some(5));
    }

    #[test]
    fn model_found_past_what_would_have_been_a_16kib_prefix_cap() {
        // A long `messages` array before `model`/`max_tokens` must not push
        // them out of reach, however it arrives in chunks.
        let filler = "a".repeat(20 * 1024);
        let prefix = format!(r#"{{"messages":[{{"content":"{filler}"}}],"#);
        let suffix = br#""model":"gpt-4","max_tokens":5}"#;
        let (model, cap) = scan(&[prefix.as_bytes(), suffix]);
        assert_eq!(model.as_deref(), Some("gpt-4"));
        assert_eq!(cap, Some(5));
    }

    #[test]
    fn utf8_character_split_across_chunk_boundary() {
        // "é" is a 2-byte UTF-8 sequence (0xC3 0xA9). Splitting the chunk
        // between those two bytes must not desync the scan; `model` and
        // `max_tokens` arrive afterward and must still resolve.
        let body = "{\"note\":\"café\",\"model\":\"gpt-4\",\"max_tokens\":5}";
        let bytes = body.as_bytes();
        let split = bytes
            .iter()
            .position(|&b| b == 0xC3)
            .map(|i| i + 1)
            .expect("fixture should contain é");
        let (model, cap) = scan(&[&bytes[..split], &bytes[split..]]);
        assert_eq!(model.as_deref(), Some("gpt-4"));
        assert_eq!(cap, Some(5));
    }

    #[test]
    fn rejects_non_integer_cap_values() {
        let (model, cap) = scan(&[br#"{"model":"gpt-4","max_tokens":1.5}"#]);
        assert_eq!(model.as_deref(), Some("gpt-4"));
        assert_eq!(cap, None);

        let (_, cap) = scan(&[br#"{"max_tokens":1e3}"#]);
        assert_eq!(cap, None);

        let (_, cap) = scan(&[br#"{"max_tokens":"100"}"#]);
        assert_eq!(cap, None);
    }

    #[test]
    fn stops_scanning_once_both_fields_resolve() {
        let mut scan = FieldScan::default();
        scan.feed_bytes(br#"{"model":"gpt-4","max_tokens":5,"#);
        assert_eq!(scan.model(), Some("gpt-4"));
        assert_eq!(scan.cap(), Some(5));
        assert!(scan.done);

        // Further bytes (even malformed ones) must not change the outcome.
        scan.feed_bytes(b"not valid json at all");
        assert_eq!(scan.model(), Some("gpt-4"));
        assert_eq!(scan.cap(), Some(5));
    }

    #[test]
    fn root_object_close_without_fields_marks_done() {
        let mut scan = FieldScan::default();
        scan.feed_bytes(br#"{"messages":[]}"#);
        assert_eq!(scan.model(), None);
        assert_eq!(scan.cap(), None);
        assert!(scan.done);
    }

    #[test]
    fn oversized_key_cannot_falsely_match_a_field_name() {
        // A key long enough to overflow KEY_CAP must never be attributed,
        // even if its first bytes happen to match a recognized field name.
        let huge_key = format!(r#""model{}""#, "x".repeat(KEY_CAP * 2));
        let body = format!(r#"{{{huge_key}:"not-a-model","model":"gpt-4","max_tokens":5}}"#);
        let (model, cap) = scan(&[body.as_bytes()]);
        assert_eq!(model.as_deref(), Some("gpt-4"));
        assert_eq!(cap, Some(5));
    }
}
