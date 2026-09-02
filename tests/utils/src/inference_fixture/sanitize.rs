// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Deterministic normalization and redaction for recorded wire fixtures.

use std::collections::BTreeMap;

use base64::{Engine as _, engine::general_purpose::STANDARD};
use http::HeaderMap;
use serde_json::Value;

use super::{
    FixtureError, FixtureProvenance, InferenceProtocol, NORMALIZATION_VERSION, RecordedBody, RecordedExchange,
    RecordedRequest, RecordedResponse, WireFixture,
    bounds::{
        MAX_SCENARIO_REQUEST_BODY_BYTES, MAX_SCRIPTED_RESPONSE_BODY_BYTES, decode_request_base64,
        decode_response_base64,
    },
    header_policy::{
        contains_configured_credential, is_credential_header, recorded_fixture_header_names, recorded_fixture_headers,
    },
};

/// Caller-provided literal replacements applied before identifier normalization.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct RedactionRules {
    /// Literal source text mapped to its safe replacement text.
    pub literals: BTreeMap<String, String>,
}

/// Maximum number of JSON-string layers decoded inside one structured carrier.
const MAX_STRUCTURED_JSON_TEXT_DEPTH: usize = 8;

/// Maximum aggregate decoded JSON text inspected inside one structured carrier.
const MAX_STRUCTURED_JSON_TEXT_BYTES: usize = MAX_SCRIPTED_RESPONSE_BODY_BYTES * 4;

/// Maximum object/array nesting inspected before Serde's recursion ceiling.
const MAX_STRUCTURED_JSON_CONTAINER_DEPTH: usize = 64;

/// Redacts sensitive fixture data and deterministically normalizes dynamic values.
///
/// A single identifier mapping is shared by every client and upstream exchange,
/// so references to the same source identifier remain linked after sanitization.
///
/// # Errors
///
/// Returns [`FixtureError::EmptyRedactionLiteral`] when a rule has an empty
/// source, or [`FixtureError::NormalizationIdOverflow`] when the normalized
/// identifier sequence is exhausted.
pub fn sanitize_fixture(fixture: &mut WireFixture, rules: &RedactionRules) -> Result<(), FixtureError> {
    Sanitizer::new(rules).sanitize_fixture(fixture)
}

/// Sanitizes once and rejects changes to stable fixture structure.
///
/// The bounded owned snapshot is required because sanitization mutably borrows
/// the fixture. It contains structural strings and header names only, never
/// request/response payloads or header values.
pub(super) fn sanitize_fixture_preserving_structure(
    fixture: &mut WireFixture,
    rules: &RedactionRules,
) -> Result<(), FixtureError> {
    let protected = ProtectedFixtureStructure::before_sanitize(fixture);
    sanitize_fixture(fixture, rules)?;
    if protected != ProtectedFixtureStructure::after_sanitize(fixture) {
        return Err(FixtureError::ReplayRuntime {
            message: "fixture sanitizer changed protected structure",
        });
    }
    Ok(())
}

/// Stable fixture structure that literal rules must not rewrite.
#[derive(Eq, PartialEq)]
struct ProtectedFixtureStructure {
    /// Wire schema version.
    version: u32,
    /// Protocol branch.
    protocol: InferenceProtocol,
    /// Stable declared scenario identifier.
    scenario_id: String,
    /// Complete source provenance identity.
    provenance: FixtureProvenance,
    /// Ordered turn and wire-boundary structure.
    turns: Vec<ProtectedTurnStructure>,
}

impl ProtectedFixtureStructure {
    /// Captures expected post-policy structure before sanitization.
    fn before_sanitize(fixture: &WireFixture) -> Self {
        Self::capture(fixture, HeaderNameMode::ExpectedCanonical)
    }

    /// Captures actual structure immediately after sanitization.
    fn after_sanitize(fixture: &WireFixture) -> Self {
        Self::capture(fixture, HeaderNameMode::Actual)
    }

    /// Copies only the bounded structural fields required across mutation.
    fn capture(fixture: &WireFixture, header_names: HeaderNameMode) -> Self {
        Self {
            version: fixture.version,
            protocol: fixture.protocol,
            scenario_id: fixture.scenario_id.clone(),
            provenance: fixture.provenance.clone(),
            turns: fixture
                .turns
                .iter()
                .map(|turn| ProtectedTurnStructure {
                    name: turn.name.clone(),
                    client: ProtectedExchangeStructure::capture(&turn.client, header_names),
                    upstream: ProtectedExchangeStructure::capture(&turn.upstream, header_names),
                })
                .collect(),
        }
    }
}

/// Whether header names are projected before or observed after sanitization.
#[derive(Clone, Copy)]
enum HeaderNameMode {
    /// Apply the sanitizer's intentional allowlist/lowercase policy.
    ExpectedCanonical,
    /// Preserve the actual post-sanitizer keys for comparison.
    Actual,
}

/// Protected structure of one ordered turn.
#[derive(Eq, PartialEq)]
struct ProtectedTurnStructure {
    /// Stable scenario turn name.
    name: String,
    /// Client-facing wire structure.
    client: ProtectedExchangeStructure,
    /// Provider-facing wire structure.
    upstream: ProtectedExchangeStructure,
}

/// Protected structure of one request/response pair.
#[derive(Eq, PartialEq)]
struct ProtectedExchangeStructure {
    /// Exact HTTP request method.
    method: String,
    /// Exact origin-form request path and query.
    path: String,
    /// Canonical request header names.
    request_header_names: Vec<String>,
    /// Request body representation and SSE framing structure.
    request_body: ProtectedBodyStructure,
    /// HTTP response status.
    status: u16,
    /// Canonical response header names.
    response_header_names: Vec<String>,
    /// Response body representation and SSE framing structure.
    response_body: ProtectedBodyStructure,
}

impl ProtectedExchangeStructure {
    /// Captures one exchange without cloning body or header values.
    fn capture(exchange: &RecordedExchange, header_names: HeaderNameMode) -> Self {
        Self {
            method: exchange.request.method.clone(),
            path: exchange.request.path.clone(),
            request_header_names: protected_header_names(&exchange.request.headers, header_names),
            request_body: ProtectedBodyStructure::capture(&exchange.request.body),
            status: exchange.response.status,
            response_header_names: protected_header_names(&exchange.response.headers, header_names),
            response_body: ProtectedBodyStructure::capture(&exchange.response.body),
        }
    }
}

/// Body variant plus SSE event framing that must remain stable.
#[derive(Eq, PartialEq)]
enum ProtectedBodyStructure {
    /// An exchange without a body.
    Empty,
    /// A structured JSON body.
    Json,
    /// A server-sent event stream.
    Sse {
        /// Whether the stream includes the terminal done marker.
        done: bool,
        /// Protected structural data for each frame, in wire order.
        frames: Vec<ProtectedSseFrameStructure>,
    },
    /// An opaque body stored as Base64.
    Base64,
}

impl ProtectedBodyStructure {
    /// Captures body shape while leaving all payload values borrowed in place.
    fn capture(body: &RecordedBody) -> Self {
        match body {
            RecordedBody::Empty => Self::Empty,
            RecordedBody::Json { .. } => Self::Json,
            RecordedBody::Sse { frames, done } => Self::Sse {
                done: *done,
                frames: frames
                    .iter()
                    .map(|frame| ProtectedSseFrameStructure {
                        event: frame.event.clone(),
                        id_present: frame.id.is_some(),
                        retry: frame.retry,
                    })
                    .collect(),
            },
            RecordedBody::Base64 { .. } => Self::Base64,
        }
    }
}

/// SSE framing fields, excluding intentionally normalized identifier values.
#[derive(Eq, PartialEq)]
struct ProtectedSseFrameStructure {
    /// Stable event type.
    event: Option<String>,
    /// Whether an identifier field exists; identifier contents may normalize.
    id_present: bool,
    /// Stable retry instruction.
    retry: Option<u64>,
}

/// Returns canonical expected or exact actual header names.
fn protected_header_names(headers: &BTreeMap<String, Vec<String>>, mode: HeaderNameMode) -> Vec<String> {
    match mode {
        HeaderNameMode::ExpectedCanonical => recorded_fixture_header_names(headers),
        HeaderNameMode::Actual => headers.keys().cloned().collect(),
    }
}

/// Verifies that a fixture has no recognizable credentials or local paths.
///
/// Validation borrows the typed fixture directly and intentionally excludes
/// fixture content from diagnostics. It reports only the violated rule and an
/// opaque JSON path so callers can safely surface the error in logs or tests.
///
/// # Errors
///
/// Returns an error when a credential header, authorization credential, or
/// absolute local path remains in the fixture.
pub fn validate_commit_safe(fixture: &WireFixture) -> Result<(), FixtureError> {
    validate_commit_safe_impl(fixture, CommitSafetyPolicy::default())
}

/// Verifies commit safety using caller-provided literal redaction rules.
///
/// Recorder and writer callers must invoke this rule-aware variant after
/// [`sanitize_fixture`] and before persisting a fixture so arbitrary configured
/// literal sources cannot be committed.
///
/// # Errors
///
/// Returns the same errors as [`validate_commit_safe`] and returns an opaque
/// error when any configured source literal remains in the fixture.
pub fn validate_commit_safe_with_rules(fixture: &WireFixture, rules: &RedactionRules) -> Result<(), FixtureError> {
    validate_commit_safe_impl(
        fixture,
        CommitSafetyPolicy {
            rules: Some(rules),
            configured_credentials: None,
        },
    )
}

/// Verifies commit safety against redaction rules and live configured credentials.
pub(super) fn validate_commit_safe_with_rules_and_credentials(
    fixture: &WireFixture,
    rules: &RedactionRules,
    configured_credentials: &HeaderMap,
) -> Result<(), FixtureError> {
    validate_commit_safe_impl(
        fixture,
        CommitSafetyPolicy {
            rules: Some(rules),
            configured_credentials: Some(configured_credentials),
        },
    )
}

/// Borrowed policy applied while walking every text-bearing fixture field.
#[derive(Clone, Copy, Default)]
struct CommitSafetyPolicy<'a> {
    /// Caller-provided literal replacements.
    rules: Option<&'a RedactionRules>,
    /// Credential headers retained only in the live recorder.
    configured_credentials: Option<&'a HeaderMap>,
}

/// Owns the mutable normalization state for one fixture sanitization pass.
struct Sanitizer<'a> {
    /// The caller-provided literal replacements.
    rules: &'a RedactionRules,
    /// The fixture-wide mapping of source IDs to normalized IDs.
    linked_ids: BTreeMap<String, String>,
    /// The next deterministic ID sequence number.
    next_identifier: u32,
    /// The current fixture JSON path.
    path: JsonPath,
}

impl<'a> Sanitizer<'a> {
    /// Creates a sanitizer with an empty fixture-wide identifier mapping.
    fn new(rules: &'a RedactionRules) -> Self {
        Self {
            rules,
            linked_ids: BTreeMap::new(),
            next_identifier: 1,
            path: JsonPath::default(),
        }
    }

    /// Sanitizes every text-bearing part of one fixture.
    fn sanitize_fixture(mut self, fixture: &mut WireFixture) -> Result<(), FixtureError> {
        validate_redaction_rules(self.rules)?;
        self.sanitize_text(&mut fixture.scenario_id);
        self.sanitize_provenance(fixture);
        self.path.push_static_key("turns");
        for (turn_index, turn) in fixture.turns.iter_mut().enumerate() {
            self.path.push_index(turn_index);
            self.sanitize_text(&mut turn.name);
            self.path.push_static_key("client");
            self.sanitize_exchange(&mut turn.client)?;
            self.path.pop();
            self.path.push_static_key("upstream");
            self.sanitize_exchange(&mut turn.upstream)?;
            self.path.pop();
            self.path.pop();
        }
        self.path.pop();
        fixture.normalization.version = NORMALIZATION_VERSION;
        fixture.normalization.linked_ids = self.linked_ids;
        Ok(())
    }

    /// Applies redaction to fixture-wide provenance strings.
    fn sanitize_provenance(&self, fixture: &mut WireFixture) {
        self.sanitize_text(&mut fixture.provenance.provider);
        self.sanitize_text(&mut fixture.provenance.model);
        if let Some(source_id) = &mut fixture.provenance.source_id {
            self.sanitize_text(source_id);
        }
    }

    /// Redacts and normalizes one client or upstream exchange.
    fn sanitize_exchange(&mut self, exchange: &mut RecordedExchange) -> Result<(), FixtureError> {
        self.path.push_static_key("request");
        self.sanitize_request(&mut exchange.request)?;
        self.path.pop();
        self.path.push_static_key("response");
        self.sanitize_response(&mut exchange.response)?;
        self.path.pop();
        Ok(())
    }

    /// Redacts and normalizes one request.
    fn sanitize_request(&mut self, request: &mut RecordedRequest) -> Result<(), FixtureError> {
        self.sanitize_text(&mut request.method);
        self.sanitize_text(&mut request.path);
        self.sanitize_headers(&mut request.headers);
        self.path.push_static_key("body");
        self.sanitize_body(&mut request.body, false)?;
        self.path.pop();
        Ok(())
    }

    /// Redacts and normalizes one response.
    fn sanitize_response(&mut self, response: &mut RecordedResponse) -> Result<(), FixtureError> {
        self.sanitize_headers(&mut response.headers);
        self.path.push_static_key("body");
        self.sanitize_body(&mut response.body, true)?;
        self.path.pop();
        Ok(())
    }

    /// Applies the header allowlist with deterministic lowercase names.
    fn sanitize_headers(&self, headers: &mut BTreeMap<String, Vec<String>>) {
        *headers = recorded_fixture_headers(std::mem::take(headers));
        for values in headers.values_mut() {
            for value in values {
                self.sanitize_text(value);
            }
        }
    }

    /// Redacts and normalizes one portable body.
    fn sanitize_body(&mut self, body: &mut RecordedBody, is_response: bool) -> Result<(), FixtureError> {
        match body {
            RecordedBody::Empty => Ok(()),
            RecordedBody::Json { value } => {
                self.path.push_static_key("value");
                if is_response {
                    self.sanitize_response_json_document(value)?;
                } else {
                    self.sanitize_json_value(value)?;
                }
                self.path.pop();
                Ok(())
            },
            RecordedBody::Sse { frames, .. } => self.sanitize_sse_frames(frames, is_response),
            RecordedBody::Base64 { data } => self.sanitize_base64(data, is_response),
        }
    }

    /// Bounded-decodes, redacts, validates binary context, and re-encodes one opaque body.
    fn sanitize_base64(&self, data: &mut String, is_response: bool) -> Result<(), FixtureError> {
        let (decoded, max_bytes, size_error) = if is_response {
            (
                decode_response_base64(data)?,
                MAX_SCRIPTED_RESPONSE_BODY_BYTES,
                "scripted response body exceeded replay limit",
            )
        } else {
            (
                decode_request_base64(data)?,
                MAX_SCENARIO_REQUEST_BODY_BYTES,
                "scenario request body exceeded replay limit",
            )
        };
        let was_binary = std::str::from_utf8(&decoded).is_err();
        let decoded = redact_decoded_bytes(decoded, self.rules, max_bytes, size_error)?;
        if was_binary && let Some(rule) = commit_safety_rule_bytes(&decoded, Some(self.rules)) {
            return Err(commit_safety_error(rule, &self.path));
        }
        *data = encode_base64_fallibly(&decoded)?;
        Ok(())
    }

    /// Redacts and normalizes every frame in one server-sent event body.
    fn sanitize_sse_frames(&mut self, frames: &mut [super::SseFrame], is_response: bool) -> Result<(), FixtureError> {
        self.path.push_static_key("frames");
        for (frame_index, frame) in frames.iter_mut().enumerate() {
            self.path.push_index(frame_index);
            if let Some(event) = &mut frame.event {
                self.sanitize_text(event);
            }
            self.path.push_static_key("data");
            self.sanitize_sse_data(&mut frame.data, is_response)?;
            self.path.pop();
            if let Some(identifier) = &mut frame.id {
                self.sanitize_text(identifier);
                self.normalize_identifier(identifier)?;
            }
            self.path.pop();
        }
        self.path.pop();
        Ok(())
    }

    /// Parses and canonicalizes JSON carried by an SSE data field when possible.
    fn sanitize_sse_data(&mut self, data: &mut String, is_response: bool) -> Result<(), FixtureError> {
        let Ok(mut value) = serde_json::from_str(data) else {
            self.sanitize_text(data);
            return Ok(());
        };
        if is_response {
            self.sanitize_response_json_document(&mut value)?;
        } else {
            self.sanitize_json_value(&mut value)?;
        }
        *data = serde_json::to_string(&value).map_err(FixtureError::JsonBodyRender)?;
        Ok(())
    }

    /// Removes unstable provider padding at a parsed response-document root.
    fn sanitize_response_json_document(&mut self, value: &mut Value) -> Result<(), FixtureError> {
        if let Value::Object(object) = value
            && (object.get("object").and_then(Value::as_str) == Some("chat.completion.chunk")
                || object.get("type").and_then(Value::as_str) == Some("response.output_text.delta"))
        {
            object.remove("obfuscation");
        }
        self.sanitize_json_value(value)
    }

    /// Recursively redacts and normalizes a JSON value using its object keys as context.
    fn sanitize_json_value(&mut self, value: &mut Value) -> Result<(), FixtureError> {
        match value {
            Value::Array(values) => {
                for (index, nested_value) in values.iter_mut().enumerate() {
                    self.path.push_index(index);
                    self.sanitize_json_value(nested_value)?;
                    self.path.pop();
                }
            },
            Value::Object(object) => self.sanitize_json_object(object)?,
            Value::String(text) => self.sanitize_text(text),
            Value::Null | Value::Bool(_) | Value::Number(_) => {},
        }
        Ok(())
    }

    /// Recursively redacts and normalizes one lexically ordered JSON object.
    fn sanitize_json_object(&mut self, object: &mut serde_json::Map<String, Value>) -> Result<(), FixtureError> {
        let is_model_object = object.get("object").and_then(Value::as_str) == Some("model");
        let sorted: BTreeMap<_, _> = std::mem::take(object).into_iter().collect();
        for (key, mut nested_value) in sorted {
            if key == "system_fingerprint" {
                continue;
            }
            self.path.push_opaque_key();
            if key == "created" {
                nested_value = Value::from(0);
            } else if key == "created_at" {
                nested_value = Value::from("1970-01-01T00:00:00Z");
            } else if key == "completed_at" && !nested_value.is_null() {
                nested_value = Value::from(0);
            } else {
                self.sanitize_json_field(&key, &mut nested_value, is_model_object)?;
            }
            self.path.pop();
            object.insert(key, nested_value);
        }
        Ok(())
    }

    /// Redacts and normalizes one JSON object field according to its key.
    fn sanitize_json_field(&mut self, key: &str, value: &mut Value, is_model_object: bool) -> Result<(), FixtureError> {
        if let Value::String(text) = value {
            if key == "arguments" {
                return self.sanitize_arguments(text);
            }
            self.sanitize_text(text);
            if is_identifier_key(key) && !(key == "id" && is_model_object) {
                self.normalize_identifier(text)?;
            }
            Ok(())
        } else {
            self.sanitize_json_value(value)
        }
    }

    /// Parses, normalizes, and serializes one JSON-encoded arguments string.
    fn sanitize_arguments(&mut self, text: &mut String) -> Result<(), FixtureError> {
        let Ok(mut arguments) = serde_json::from_str(text) else {
            self.sanitize_text(text);
            return Ok(());
        };
        self.sanitize_json_value(&mut arguments)?;
        canonicalize_json(&mut arguments);
        *text = serde_json::to_string(&arguments).map_err(FixtureError::JsonBodyRender)?;
        Ok(())
    }

    /// Replaces a recognized provider identifier while preserving repeated references.
    fn normalize_identifier(&mut self, identifier: &mut String) -> Result<(), FixtureError> {
        let Some(prefix) = identifier_prefix(identifier) else {
            return Ok(());
        };
        if let Some(normalized) = self.linked_ids.get(identifier) {
            identifier.clone_from(normalized);
            return Ok(());
        }
        let current = self.next_identifier;
        self.next_identifier = self
            .next_identifier
            .checked_add(1)
            .ok_or(FixtureError::NormalizationIdOverflow)?;
        let normalized = format!("{prefix}{current:04}");
        let source = std::mem::replace(identifier, normalized.clone());
        self.linked_ids.insert(source, normalized);
        Ok(())
    }

    /// Applies each rule once to its original string input without cascading replacements.
    fn sanitize_text(&self, text: &mut String) {
        if self.rules.literals.is_empty() {
            return;
        }
        *text = redact_text(text, self.rules);
    }
}

/// Sorts all JSON object keys recursively for canonical argument serialization.
fn canonicalize_json(value: &mut Value) {
    match value {
        Value::Array(values) => {
            for nested_value in values {
                canonicalize_json(nested_value);
            }
        },
        Value::Object(object) => {
            let original = std::mem::take(object);
            let mut sorted = BTreeMap::new();
            for (key, mut nested_value) in original {
                canonicalize_json(&mut nested_value);
                sorted.insert(key, nested_value);
            }
            object.extend(sorted);
        },
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) => {},
    }
}

/// Rejects redaction rules whose empty source would match every string position.
fn validate_redaction_rules(rules: &RedactionRules) -> Result<(), FixtureError> {
    if rules.literals.keys().any(String::is_empty) {
        Err(FixtureError::EmptyRedactionLiteral)
    } else {
        Ok(())
    }
}

/// Replaces matching sources from the original input using leftmost-longest order.
fn redact_text(text: &str, rules: &RedactionRules) -> String {
    let mut redacted = String::with_capacity(text.len());
    let mut offset = 0;
    while offset < text.len() {
        let Some(remainder) = text.get(offset..) else {
            return redacted;
        };
        let replacement = longest_matching_replacement(remainder, rules);
        if let Some((source, replacement)) = replacement {
            redacted.push_str(replacement);
            offset += source.len();
        } else if let Some(character) = remainder.chars().next() {
            redacted.push(character);
            offset += character.len_utf8();
        }
    }
    redacted
}

/// Returns the longest matching source, resolving theoretical ties lexicographically.
fn longest_matching_replacement<'a>(text: &str, rules: &'a RedactionRules) -> Option<(&'a str, &'a str)> {
    let mut selected = None;
    for (source, replacement) in &rules.literals {
        if text.starts_with(source) {
            let replace_selected = selected.is_none_or(|(selected_source, _): (&str, &str)| {
                source.len() > selected_source.len()
                    || (source.len() == selected_source.len() && source.as_str() < selected_source)
            });
            if replace_selected {
                selected = Some((source.as_str(), replacement.as_str()));
            }
        }
    }
    selected
}

/// Applies text rules to owned decoded bytes without interpreting invalid UTF-8.
fn redact_decoded_bytes(
    decoded: Vec<u8>,
    rules: &RedactionRules,
    max_bytes: usize,
    size_error: &'static str,
) -> Result<Vec<u8>, FixtureError> {
    if rules.literals.is_empty()
        || !rules
            .literals
            .keys()
            .any(|source| contains_bytes(&decoded, source.as_bytes()))
    {
        return Ok(decoded);
    }

    let mut redacted = Vec::new();
    redacted
        .try_reserve_exact(decoded.len().min(max_bytes))
        .map_err(|_source| runtime_error("recorded Base64 body allocation failed"))?;
    let mut offset = 0;
    while offset < decoded.len() {
        let remainder = &decoded[offset..];
        if let Some((source, replacement)) = longest_matching_byte_replacement(remainder, rules) {
            extend_decoded_output(&mut redacted, replacement, max_bytes, size_error)?;
            offset += source.len();
        } else {
            extend_decoded_output(&mut redacted, &remainder[..1], max_bytes, size_error)?;
            offset += 1;
        }
    }
    Ok(redacted)
}

/// Returns the longest matching UTF-8 rule as byte slices.
fn longest_matching_byte_replacement<'a>(bytes: &[u8], rules: &'a RedactionRules) -> Option<(&'a [u8], &'a [u8])> {
    let mut selected = None;
    for (source, replacement) in &rules.literals {
        let source = source.as_bytes();
        if bytes.starts_with(source) {
            let replace_selected = selected.is_none_or(|(selected_source, _): (&[u8], &[u8])| {
                source.len() > selected_source.len()
                    || (source.len() == selected_source.len() && source < selected_source)
            });
            if replace_selected {
                selected = Some((source, replacement.as_bytes()));
            }
        }
    }
    selected
}

/// Extends a decoded redaction output only after size and allocation checks.
fn extend_decoded_output(
    output: &mut Vec<u8>,
    bytes: &[u8],
    max_bytes: usize,
    size_error: &'static str,
) -> Result<(), FixtureError> {
    let total = output
        .len()
        .checked_add(bytes.len())
        .ok_or_else(|| runtime_error(size_error))?;
    if total > max_bytes {
        return Err(runtime_error(size_error));
    }
    output
        .try_reserve_exact(bytes.len())
        .map_err(|_source| runtime_error("recorded Base64 body allocation failed"))?;
    output.extend_from_slice(bytes);
    Ok(())
}

/// Re-encodes bounded decoded bytes after fallibly reserving the exact output size.
fn encode_base64_fallibly(decoded: &[u8]) -> Result<String, FixtureError> {
    let encoded_len = decoded
        .len()
        .div_ceil(3)
        .checked_mul(4)
        .ok_or_else(|| runtime_error("recorded Base64 body allocation failed"))?;
    let mut encoded = String::new();
    encoded
        .try_reserve_exact(encoded_len)
        .map_err(|_source| runtime_error("recorded Base64 body allocation failed"))?;
    STANDARD.encode_string(decoded, &mut encoded);
    Ok(encoded)
}

/// Returns whether a JSON field name identifies a provider identifier.
fn is_identifier_key(key: &str) -> bool {
    matches!(
        key,
        "id" | "response_id" | "previous_response_id" | "call_id" | "tool_call_id" | "conversation_id" | "item_id"
    )
}

/// Returns the canonical replacement prefix for a recognized provider identifier.
fn identifier_prefix(identifier: &str) -> Option<&'static str> {
    [
        ("resp_", "resp_recorded_"),
        ("msg_", "msg_recorded_"),
        ("chatcmpl-", "chatcmpl-recorded-"),
        ("call_", "call_recorded_"),
        ("toolu_", "toolu_recorded_"),
        ("conv_", "conv_recorded_"),
    ]
    .into_iter()
    .find_map(|(source_prefix, normalized_prefix)| identifier.starts_with(source_prefix).then_some(normalized_prefix))
}

/// Borrows and scans every text-bearing field in the typed fixture.
fn validate_commit_safe_impl(fixture: &WireFixture, policy: CommitSafetyPolicy<'_>) -> Result<(), FixtureError> {
    if let Some(rules) = policy.rules {
        validate_redaction_rules(rules)?;
    }
    let mut path = JsonPath::default();
    validate_field_text(&mut path, "scenario_id", &fixture.scenario_id, policy)?;
    validate_provenance(&mut path, &fixture.provenance, policy)?;
    validate_normalization(&mut path, &fixture.normalization.linked_ids, policy)?;
    path.push_static_key("turns");
    for (index, turn) in fixture.turns.iter().enumerate() {
        path.push_index(index);
        validate_field_text(&mut path, "name", &turn.name, policy)?;
        path.push_static_key("client");
        validate_exchange(&turn.client, policy, &mut path)?;
        path.pop();
        path.push_static_key("upstream");
        validate_exchange(&turn.upstream, policy, &mut path)?;
        path.pop();
        path.pop();
    }
    path.pop();
    Ok(())
}

/// Scans source provenance without copying its strings.
fn validate_provenance(
    path: &mut JsonPath,
    provenance: &FixtureProvenance,
    policy: CommitSafetyPolicy<'_>,
) -> Result<(), FixtureError> {
    path.push_static_key("provenance");
    validate_field_text(path, "provider", &provenance.provider, policy)?;
    validate_field_text(path, "model", &provenance.model, policy)?;
    if let Some(source_id) = &provenance.source_id {
        validate_field_text(path, "source_id", source_id, policy)?;
    }
    path.pop();
    Ok(())
}

/// Scans both keys and values in the linked-identifier map.
fn validate_normalization(
    path: &mut JsonPath,
    linked_ids: &BTreeMap<String, String>,
    policy: CommitSafetyPolicy<'_>,
) -> Result<(), FixtureError> {
    path.push_static_key("normalization");
    path.push_static_key("linked_ids");
    for (source, normalized) in linked_ids {
        path.push_opaque_key();
        validate_text(source, policy, path)?;
        validate_text(normalized, policy, path)?;
        path.pop();
    }
    path.pop();
    path.pop();
    Ok(())
}

/// Scans one typed request/response exchange.
fn validate_exchange(
    exchange: &RecordedExchange,
    policy: CommitSafetyPolicy<'_>,
    path: &mut JsonPath,
) -> Result<(), FixtureError> {
    path.push_static_key("request");
    validate_field_text(path, "method", &exchange.request.method, policy)?;
    validate_field_text(path, "path", &exchange.request.path, policy)?;
    validate_headers(path, &exchange.request.headers, policy)?;
    validate_body(path, &exchange.request.body, policy, false)?;
    path.pop();
    path.push_static_key("response");
    validate_headers(path, &exchange.response.headers, policy)?;
    validate_body(path, &exchange.response.body, policy, true)?;
    path.pop();
    Ok(())
}

/// Scans one real wire header map with credential-name context.
fn validate_headers(
    path: &mut JsonPath,
    headers: &BTreeMap<String, Vec<String>>,
    policy: CommitSafetyPolicy<'_>,
) -> Result<(), FixtureError> {
    path.push_static_key("headers");
    for (name, values) in headers {
        path.push_opaque_key();
        if is_credential_header(name) {
            let error = commit_safety_error("credential header", path);
            path.pop();
            path.pop();
            return Err(error);
        }
        validate_text(name, policy, path)?;
        for (index, value) in values.iter().enumerate() {
            path.push_index(index);
            validate_text(value, policy, path)?;
            path.pop();
        }
        path.pop();
    }
    path.pop();
    Ok(())
}

/// Scans one typed body, including SSE metadata and arbitrary JSON payloads.
fn validate_body(
    path: &mut JsonPath,
    body: &RecordedBody,
    policy: CommitSafetyPolicy<'_>,
    is_response: bool,
) -> Result<(), FixtureError> {
    path.push_static_key("body");
    match body {
        RecordedBody::Empty => {},
        RecordedBody::Json { value } => {
            path.push_static_key("value");
            validate_json_value(value, policy, path)?;
            path.pop();
        },
        RecordedBody::Sse { frames, .. } => {
            path.push_static_key("frames");
            for (index, frame) in frames.iter().enumerate() {
                path.push_index(index);
                if let Some(event) = &frame.event {
                    validate_field_text(path, "event", event, policy)?;
                }
                path.push_static_key("data");
                validate_nested_json_text(&frame.data, policy, path)?;
                path.pop();
                if let Some(id) = &frame.id {
                    validate_field_text(path, "id", id, policy)?;
                }
                path.pop();
            }
            path.pop();
        },
        RecordedBody::Base64 { data } => validate_base64_field(path, data, policy, is_response)?,
    }
    path.pop();
    Ok(())
}

/// Adds opaque body-path context around bounded decoded-byte validation.
fn validate_base64_field(
    path: &mut JsonPath,
    data: &str,
    policy: CommitSafetyPolicy<'_>,
    is_response: bool,
) -> Result<(), FixtureError> {
    path.push_static_key("data");
    let result = validate_base64_body(data, policy, is_response, path);
    path.pop();
    result
}

/// Bounded-decodes and validates one opaque body without exposing its bytes.
fn validate_base64_body(
    data: &str,
    policy: CommitSafetyPolicy<'_>,
    is_response: bool,
    path: &JsonPath,
) -> Result<(), FixtureError> {
    let decoded = if is_response {
        decode_response_base64(data)?
    } else {
        decode_request_base64(data)?
    };
    validate_bytes(&decoded, policy, path)
}

/// Pushes a fixed schema field while scanning one borrowed string.
fn validate_field_text(
    path: &mut JsonPath,
    field: &'static str,
    text: &str,
    policy: CommitSafetyPolicy<'_>,
) -> Result<(), FixtureError> {
    path.push_static_key(field);
    let result = validate_text(text, policy, path);
    path.pop();
    result
}

/// Scans one borrowed body JSON value and produces an opaque safety error.
fn validate_json_value(value: &Value, policy: CommitSafetyPolicy<'_>, path: &mut JsonPath) -> Result<(), FixtureError> {
    match value {
        Value::Array(values) => {
            for (index, nested_value) in values.iter().enumerate() {
                path.push_index(index);
                validate_json_value(nested_value, policy, path)?;
                path.pop();
            }
        },
        Value::Object(object) => validate_json_object(object, policy, path)?,
        Value::String(text) => validate_text(text, policy, path)?,
        Value::Null | Value::Bool(_) | Value::Number(_) => {},
    }
    Ok(())
}

/// Scans one JSON object in lexical key order for deterministic diagnostics.
fn validate_json_object(
    object: &serde_json::Map<String, Value>,
    policy: CommitSafetyPolicy<'_>,
    path: &mut JsonPath,
) -> Result<(), FixtureError> {
    let mut keys: Vec<_> = object.keys().collect();
    keys.sort_unstable();
    for key in keys {
        let Some(value) = object.get(key) else {
            continue;
        };
        if path.hides_object_keys() {
            path.push_opaque_key();
        } else {
            path.push_static_key(key);
        }
        validate_text(key, policy, path)?;
        if key == "arguments"
            && let Value::String(text) = value
        {
            validate_nested_json_text(text, policy, path)?;
        } else {
            validate_json_value(value, policy, path)?;
        }
        path.pop();
    }
    Ok(())
}

/// Validates one string and recursively scans it when it carries JSON text.
fn validate_nested_json_text(
    text: &str,
    policy: CommitSafetyPolicy<'_>,
    path: &mut JsonPath,
) -> Result<(), FixtureError> {
    let mut remaining_bytes = MAX_STRUCTURED_JSON_TEXT_BYTES;
    validate_nested_json_text_bounded(text, policy, path, 0, &mut remaining_bytes)
}

/// Recursively decodes structured strings with explicit depth and byte budgets.
fn validate_nested_json_text_bounded(
    text: &str,
    policy: CommitSafetyPolicy<'_>,
    path: &mut JsonPath,
    depth: usize,
    remaining_bytes: &mut usize,
) -> Result<(), FixtureError> {
    validate_text(text, policy, path)?;
    *remaining_bytes = remaining_bytes
        .checked_sub(text.len())
        .ok_or_else(|| commit_safety_error("structured JSON text byte limit", path))?;
    validate_nested_json_container_depth(text, path)?;
    let Ok(value) = serde_json::from_str(text) else {
        return Ok(());
    };
    if depth >= MAX_STRUCTURED_JSON_TEXT_DEPTH {
        return Err(commit_safety_error("structured JSON text depth limit", path));
    }
    validate_nested_json_value(&value, policy, path, depth + 1, remaining_bytes)
}

/// Rejects pathological structural nesting without counting delimiters in strings.
fn validate_nested_json_container_depth(text: &str, path: &JsonPath) -> Result<(), FixtureError> {
    let mut depth = 0_usize;
    let mut in_string = false;
    let mut escaped = false;
    for byte in text.bytes() {
        if in_string {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                in_string = false;
            }
            continue;
        }
        match byte {
            b'"' => in_string = true,
            b'{' | b'[' => {
                depth = depth.saturating_add(1);
                if depth > MAX_STRUCTURED_JSON_CONTAINER_DEPTH {
                    return Err(commit_safety_error("structured JSON container depth limit", path));
                }
            },
            b'}' | b']' => depth = depth.saturating_sub(1),
            _ => {},
        }
    }
    Ok(())
}

/// Traverses decoded carrier JSON and attempts another decode for every string.
fn validate_nested_json_value(
    value: &Value,
    policy: CommitSafetyPolicy<'_>,
    path: &mut JsonPath,
    depth: usize,
    remaining_bytes: &mut usize,
) -> Result<(), FixtureError> {
    match value {
        Value::Array(values) => {
            for (index, nested_value) in values.iter().enumerate() {
                path.push_index(index);
                validate_nested_json_value(nested_value, policy, path, depth, remaining_bytes)?;
                path.pop();
            }
        },
        Value::Object(object) => validate_nested_json_object(object, policy, path, depth, remaining_bytes)?,
        Value::String(nested_text) => {
            validate_nested_json_text_bounded(nested_text, policy, path, depth, remaining_bytes)?;
        },
        Value::Null | Value::Bool(_) | Value::Number(_) => {},
    }
    Ok(())
}

/// Traverses one decoded carrier object in deterministic key order.
fn validate_nested_json_object(
    object: &serde_json::Map<String, Value>,
    policy: CommitSafetyPolicy<'_>,
    path: &mut JsonPath,
    depth: usize,
    remaining_bytes: &mut usize,
) -> Result<(), FixtureError> {
    let mut keys: Vec<_> = object.keys().collect();
    keys.sort_unstable();
    for key in keys {
        let Some(nested_value) = object.get(key) else {
            continue;
        };
        if path.hides_object_keys() {
            path.push_opaque_key();
        } else {
            path.push_static_key(key);
        }
        validate_text(key, policy, path)?;
        validate_nested_json_value(nested_value, policy, path, depth, remaining_bytes)?;
        path.pop();
    }
    Ok(())
}

/// Rejects unsafe text with a path rendered only when reporting the error.
fn validate_text(text: &str, policy: CommitSafetyPolicy<'_>, path: &JsonPath) -> Result<(), FixtureError> {
    if let Some(rule) = commit_safety_policy_rule(text.as_bytes(), policy) {
        return Err(commit_safety_error(rule, path));
    }
    Ok(())
}

/// Returns the first generic commit-safety rule violated by decoded text.
pub(super) fn commit_safety_rule(text: &str, rules: Option<&RedactionRules>) -> Option<&'static str> {
    commit_safety_rule_bytes(text.as_bytes(), rules)
}

/// Returns the first generic commit-safety rule violated by decoded bytes.
fn commit_safety_rule_bytes(bytes: &[u8], rules: Option<&RedactionRules>) -> Option<&'static str> {
    if rules.is_some_and(|rules| {
        rules
            .literals
            .keys()
            .any(|source| contains_bytes(bytes, source.as_bytes()))
    }) {
        Some("unredacted literal")
    } else if contains_authorization_scheme(bytes, b"Bearer") {
        Some("bearer token")
    } else if contains_authorization_scheme(bytes, b"Basic") {
        Some("basic authentication")
    } else if contains_windows_absolute_path(bytes) {
        Some("Windows drive path")
    } else if contains_posix_local_path(bytes, b"/Users/") {
        Some("absolute user path")
    } else if contains_posix_local_path(bytes, b"/home/") {
        Some("absolute home path")
    } else {
        None
    }
}

/// Returns the first configured or generic commit-safety rule violated.
fn commit_safety_policy_rule(bytes: &[u8], policy: CommitSafetyPolicy<'_>) -> Option<&'static str> {
    if policy
        .configured_credentials
        .is_some_and(|configured| contains_configured_credential(configured, bytes))
    {
        Some("configured credential")
    } else {
        commit_safety_rule_bytes(bytes, policy.rules)
    }
}

/// Applies commit-safety matching to an arbitrary decoded byte slice.
fn validate_bytes(bytes: &[u8], policy: CommitSafetyPolicy<'_>, path: &JsonPath) -> Result<(), FixtureError> {
    if let Some(rule) = commit_safety_policy_rule(bytes, policy) {
        return Err(commit_safety_error(rule, path));
    }
    Ok(())
}

/// Constructs a commit-safety error without including fixture content.
fn commit_safety_error(rule: &'static str, path: &JsonPath) -> FixtureError {
    FixtureError::CommitSafety {
        rule,
        path: path.render(),
    }
}

/// Returns whether decoded bytes contain one HTTP authorization scheme and whitespace.
fn contains_authorization_scheme(bytes: &[u8], scheme: &[u8]) -> bool {
    bytes.windows(scheme.len() + 1).any(|candidate| {
        candidate[..scheme.len()].eq_ignore_ascii_case(scheme) && is_ascii_whitespace(candidate[scheme.len()])
    })
}

/// Returns whether decoded bytes contain a local POSIX path at a lexical boundary.
fn contains_posix_local_path(bytes: &[u8], prefix: &[u8]) -> bool {
    bytes.windows(prefix.len()).enumerate().any(|(index, candidate)| {
        candidate == prefix
            && !is_http_url_path(bytes, index)
            && (is_file_url_path(bytes, index) || is_lexical_boundary(bytes, index))
    })
}

/// Returns whether decoded bytes contain a drive-qualified absolute Windows path.
fn contains_windows_absolute_path(bytes: &[u8]) -> bool {
    bytes.windows(3).enumerate().any(|(index, candidate)| {
        candidate[0].is_ascii_alphabetic()
            && candidate[1] == b':'
            && matches!(candidate[2], b'\\' | b'/')
            && !is_http_url_path(bytes, index)
            && (is_file_url_path(bytes, index) || is_lexical_boundary(bytes, index))
    })
}

/// Returns whether a candidate begins at a lexical token boundary.
fn is_lexical_boundary(bytes: &[u8], index: usize) -> bool {
    index == 0 || is_ascii_whitespace(bytes[index - 1]) || !is_identifier_byte(bytes[index - 1])
}

/// Returns whether one byte is one of the six ASCII whitespace code points.
fn is_ascii_whitespace(byte: u8) -> bool {
    matches!(byte, b'\t' | b'\n' | 0x0B | 0x0C | b'\r' | b' ')
}

/// Returns whether one byte can continue an ordinary identifier token.
fn is_identifier_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-')
}

/// Returns whether a local-looking path candidate belongs to an HTTP(S) URL path.
fn is_http_url_path(bytes: &[u8], candidate_index: usize) -> bool {
    url_path_context(bytes, candidate_index).is_some_and(|scheme| matches!(scheme, UrlScheme::Http))
}

/// Returns whether a local-looking path candidate belongs to a local `file:` URL.
fn is_file_url_path(bytes: &[u8], candidate_index: usize) -> bool {
    url_path_context(bytes, candidate_index).is_some_and(|scheme| matches!(scheme, UrlScheme::File))
}

/// Supported URL schemes used only to distinguish remote path components from local URLs.
#[derive(Clone, Copy)]
enum UrlScheme {
    /// An HTTP or HTTPS URL, whose path component is ordinary remote data.
    Http,
    /// A file URL, whose path component remains a local path.
    File,
}

/// Returns the URL scheme when a candidate lies at or below that URL's path start.
fn url_path_context(bytes: &[u8], candidate_index: usize) -> Option<UrlScheme> {
    let token_start = bytes[..candidate_index]
        .iter()
        .rposition(|byte| is_url_token_boundary(*byte))
        .map_or(0, |index| index + 1);
    let token = &bytes[token_start..];
    let (scheme, authority_start) = if starts_with_ignore_ascii_case(token, b"http://") {
        (UrlScheme::Http, b"http://".len())
    } else if starts_with_ignore_ascii_case(token, b"https://") {
        (UrlScheme::Http, b"https://".len())
    } else if starts_with_ignore_ascii_case(token, b"file://") {
        (UrlScheme::File, b"file://".len())
    } else {
        return None;
    };
    let relative_candidate = candidate_index.checked_sub(token_start)?;
    let path_start = token
        .get(authority_start..)?
        .iter()
        .position(|byte| matches!(byte, b'/' | b'?' | b'#'))?
        .checked_add(authority_start)?;
    if path_start == authority_start || token[path_start] != b'/' {
        return None;
    }
    let path_end = token[path_start..]
        .iter()
        .position(|byte| matches!(byte, b'?' | b'#'))
        .and_then(|offset| path_start.checked_add(offset))
        .unwrap_or(token.len());
    (relative_candidate >= path_start && relative_candidate < path_end).then_some(scheme)
}

/// Returns whether a byte terminates the simple URL token used by safety matching.
fn is_url_token_boundary(byte: u8) -> bool {
    is_ascii_whitespace(byte) || matches!(byte, b'"' | b'\'' | b'`' | b'<' | b'>' | b'(' | b')' | b'{' | b'}')
}

/// Returns whether a byte slice begins with an ASCII case-insensitive prefix.
fn starts_with_ignore_ascii_case(bytes: &[u8], prefix: &[u8]) -> bool {
    bytes
        .get(..prefix.len())
        .is_some_and(|candidate| candidate.eq_ignore_ascii_case(prefix))
}

/// Returns whether a byte slice contains another nonempty byte slice.
fn contains_bytes(haystack: &[u8], needle: &[u8]) -> bool {
    !needle.is_empty() && haystack.windows(needle.len()).any(|candidate| candidate == needle)
}

/// Creates an opaque replay validation error.
fn runtime_error(message: &'static str) -> FixtureError {
    FixtureError::ReplayRuntime { message }
}

/// A mutable JSON path segment stack used during recursive fixture processing.
#[derive(Default)]
struct JsonPath {
    /// Ordered JSON Pointer segments from the fixture root.
    segments: Vec<JsonPathSegment>,
}

/// One rendered JSON Pointer segment that either preserves schema context or hides data keys.
enum JsonPathSegment {
    /// A fixed field name from the wire-fixture schema.
    Static(String),
    /// A stable array index.
    Index(usize),
    /// An arbitrary JSON object or header name that must not appear in diagnostics.
    OpaqueKey,
}

impl JsonPath {
    /// Pushes one fixed wire-fixture schema field name.
    fn push_static_key(&mut self, key: &str) {
        self.segments.push(JsonPathSegment::Static(key.to_owned()));
    }

    /// Pushes an arbitrary object key without retaining its contents.
    fn push_opaque_key(&mut self) {
        self.segments.push(JsonPathSegment::OpaqueKey);
    }

    /// Pushes one array-index segment.
    fn push_index(&mut self, index: usize) {
        self.segments.push(JsonPathSegment::Index(index));
    }

    /// Removes the current leaf segment after its recursive visit finishes.
    fn pop(&mut self) {
        let _segment = self.segments.pop();
    }

    /// Renders the stack as an escaped JSON Pointer rooted at `$`.
    fn render(&self) -> String {
        let mut path = String::from("$");
        for segment in &self.segments {
            path.push('/');
            let segment = match segment {
                JsonPathSegment::Static(segment) => segment.as_str(),
                JsonPathSegment::Index(index) => {
                    let rendered = index.to_string();
                    path.push_str(&rendered);
                    continue;
                },
                JsonPathSegment::OpaqueKey => "<key>",
            };
            for character in segment.chars() {
                match character {
                    '~' => path.push_str("~0"),
                    '/' => path.push_str("~1"),
                    _ => path.push(character),
                }
            }
        }
        path
    }

    /// Returns whether arbitrary object keys must be hidden at the current schema location.
    fn hides_object_keys(&self) -> bool {
        self.is_wire_header_map()
            || self.is_body_value_or_descendant()
            || self.is_sse_data_or_descendant()
            || self.is_linked_identifier_map()
            || self.contains_opaque_key()
    }

    /// Returns whether this path contains an arbitrary object key.
    fn contains_opaque_key(&self) -> bool {
        self.segments
            .iter()
            .any(|segment| matches!(segment, JsonPathSegment::OpaqueKey))
    }

    /// Returns whether the current path is a recorded JSON body's value or a descendant.
    fn is_body_value_or_descendant(&self) -> bool {
        matches!(
            self.segments.as_slice(),
            [
                JsonPathSegment::Static(turns),
                JsonPathSegment::Index(_),
                JsonPathSegment::Static(side),
                JsonPathSegment::Static(exchange),
                JsonPathSegment::Static(body),
                JsonPathSegment::Static(value),
                ..
            ] if turns == "turns"
                && matches!(side.as_str(), "client" | "upstream")
                && matches!(exchange.as_str(), "request" | "response")
                && body == "body"
                && value == "value"
        )
    }

    /// Returns whether the current path is parsed JSON inside one SSE data field.
    fn is_sse_data_or_descendant(&self) -> bool {
        matches!(
            self.segments.as_slice(),
            [
                JsonPathSegment::Static(turns),
                JsonPathSegment::Index(_),
                JsonPathSegment::Static(side),
                JsonPathSegment::Static(exchange),
                JsonPathSegment::Static(body),
                JsonPathSegment::Static(frames),
                JsonPathSegment::Index(_),
                JsonPathSegment::Static(data),
                ..
            ] if turns == "turns"
                && matches!(side.as_str(), "client" | "upstream")
                && matches!(exchange.as_str(), "request" | "response")
                && body == "body"
                && frames == "frames"
                && data == "data"
        )
    }

    /// Returns whether the current path is the normalization map keyed by source identifiers.
    fn is_linked_identifier_map(&self) -> bool {
        matches!(
            self.segments.as_slice(),
            [JsonPathSegment::Static(normalization), JsonPathSegment::Static(linked_ids)]
                if normalization == "normalization" && linked_ids == "linked_ids"
        )
    }

    /// Returns whether the current path identifies an exchange's actual header map at the fixture root.
    fn is_wire_header_map(&self) -> bool {
        matches!(
            self.segments.as_slice(),
            [
                JsonPathSegment::Static(turns),
                JsonPathSegment::Index(_),
                JsonPathSegment::Static(side),
                JsonPathSegment::Static(exchange),
                JsonPathSegment::Static(headers),
            ] if turns == "turns"
                && matches!(side.as_str(), "client" | "upstream")
                && matches!(exchange.as_str(), "request" | "response")
                && headers == "headers"
        )
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use base64::{Engine as _, engine::general_purpose::STANDARD};
    use http::{HeaderMap, HeaderValue};
    use serde_json::{Value, json};

    use super::{
        RedactionRules, sanitize_fixture, validate_commit_safe, validate_commit_safe_with_rules,
        validate_commit_safe_with_rules_and_credentials,
    };
    use crate::inference_fixture::{
        FixtureProvenance, InferenceProtocol, NormalizationMetadata, ProvenanceKind, RecordedBody, RecordedExchange,
        RecordedRequest, RecordedResponse, WireFixture, WireTurn,
    };

    fn exchange(
        request_headers: BTreeMap<String, Vec<String>>,
        request_value: Value,
        response_headers: BTreeMap<String, Vec<String>>,
        response_value: Value,
    ) -> RecordedExchange {
        RecordedExchange {
            request: RecordedRequest {
                method: "POST".to_owned(),
                path: "/v1/responses".to_owned(),
                headers: request_headers,
                body: RecordedBody::Json { value: request_value },
            },
            response: RecordedResponse {
                status: 200,
                headers: response_headers,
                body: RecordedBody::Json { value: response_value },
            },
        }
    }

    fn headers(entries: &[(&str, &[&str])]) -> BTreeMap<String, Vec<String>> {
        entries
            .iter()
            .map(|(name, values)| {
                (
                    (*name).to_owned(),
                    values.iter().map(|value| (*value).to_owned()).collect(),
                )
            })
            .collect()
    }

    fn fixture() -> WireFixture {
        let provenance = FixtureProvenance {
            kind: ProvenanceKind::Live,
            provider: "provider".to_owned(),
            model: "model-under-test".to_owned(),
            source_id: Some("capture".to_owned()),
        };
        let turn = WireTurn {
            name: "turn".to_owned(),
            client: client_exchange(),
            upstream: upstream_exchange(),
        };
        WireFixture {
            version: 1,
            scenario_id: "sanitize".to_owned(),
            protocol: InferenceProtocol::OpenaiResponses,
            provenance,
            normalization: NormalizationMetadata {
                version: 0,
                linked_ids: BTreeMap::new(),
            },
            turns: vec![turn],
        }
    }

    fn client_exchange() -> RecordedExchange {
        exchange(
            client_request_headers(),
            client_request_value(),
            client_response_headers(),
            client_response_value(),
        )
    }

    fn client_request_headers() -> BTreeMap<String, Vec<String>> {
        headers(&[
            ("Authorization", &["Bearer client-secret"]),
            ("Content-Type", &["application/json", "application/problem+json"]),
            ("X-Ignored", &["ignored"]),
        ])
    }

    fn client_request_value() -> Value {
        json!({
            "response_id": "resp_source",
            "call_id": "call_source",
            "model": "resp_model_request",
            "unrelated_id": "resp_not_an_identifier",
            "arguments": "{\"z\":1,\"a\":{\"y\":2,\"x\":3}}",
        })
    }

    fn client_response_headers() -> BTreeMap<String, Vec<String>> {
        headers(&[
            ("Proxy-Authorization", &["Bearer response-secret"]),
            ("Request-ID", &["request-1"]),
        ])
    }

    fn client_response_value() -> Value {
        json!({
            "id": "resp_source",
            "previous_response_id": "resp_previous",
            "created": 123,
            "created_at": "2026-08-04T12:00:00Z",
            "system_fingerprint": "fp_secret",
        })
    }

    fn upstream_exchange() -> RecordedExchange {
        exchange(
            upstream_request_headers(),
            upstream_request_value(),
            upstream_response_headers(),
            json!({"id": "resp_previous", "previous_response_id": "literal-source"}),
        )
    }

    fn upstream_request_headers() -> BTreeMap<String, Vec<String>> {
        headers(&[
            ("Cookie", &["session=secret"]),
            ("anthropic-version", &["2023-06-01"]),
            ("x-api-key", &["upstream-key"]),
        ])
    }

    fn upstream_request_value() -> Value {
        json!({
            "id": "resp_source",
            "tool_call_id": "call_source",
            "conversation_id": "conv_source",
            "nested": [
                {"id": "msg_source"}, {"id": "chatcmpl-source"},
                {"id": "toolu_source"}, {"object": "model", "id": "resp_model_object"},
            ],
        })
    }

    fn upstream_response_headers() -> BTreeMap<String, Vec<String>> {
        headers(&[
            ("Set-Cookie", &["a=secret"]),
            ("API-Key", &["api-key"]),
            ("openai-beta", &["responses=experimental"]),
            ("X-Request-ID", &["request-2"]),
        ])
    }

    #[test]
    fn sanitize_removes_sensitive_headers_and_preserves_only_allowlisted_lowercase_values() {
        // Arrange
        let mut fixture = fixture();
        let rules = RedactionRules::default();

        // Act
        sanitize_fixture(&mut fixture, &rules).unwrap();

        // Assert
        assert_eq!(
            fixture.turns[0].client.request.headers,
            headers(&[("content-type", &["application/json", "application/problem+json"])])
        );
        assert_eq!(
            fixture.turns[0].client.response.headers,
            headers(&[("request-id", &["request-1"])])
        );
        assert_eq!(
            fixture.turns[0].upstream.request.headers,
            headers(&[("anthropic-version", &["2023-06-01"])])
        );
        assert_eq!(
            fixture.turns[0].upstream.response.headers,
            headers(&[
                ("openai-beta", &["responses=experimental"]),
                ("x-request-id", &["request-2"])
            ])
        );
    }

    #[test]
    fn sanitize_links_known_ids_in_first_seen_order_without_rewriting_models_or_non_target_fields() {
        // Arrange
        let mut fixture = fixture();
        let rules = RedactionRules::default();

        // Act
        sanitize_fixture(&mut fixture, &rules).unwrap();

        // Assert
        assert_expected_linked_ids(&fixture);
        assert_normalized_request_ids(&fixture);
    }

    fn assert_expected_linked_ids(fixture: &WireFixture) {
        assert_eq!(
            fixture.normalization.linked_ids,
            BTreeMap::from([
                ("call_source".to_owned(), "call_recorded_0001".to_owned()),
                ("chatcmpl-source".to_owned(), "chatcmpl-recorded-0006".to_owned()),
                ("conv_source".to_owned(), "conv_recorded_0004".to_owned()),
                ("msg_source".to_owned(), "msg_recorded_0005".to_owned()),
                ("resp_previous".to_owned(), "resp_recorded_0003".to_owned()),
                ("resp_source".to_owned(), "resp_recorded_0002".to_owned()),
                ("toolu_source".to_owned(), "toolu_recorded_0007".to_owned()),
            ])
        );
    }

    fn assert_normalized_request_ids(fixture: &WireFixture) {
        let client_request = &fixture.turns[0].client.request.body;
        let RecordedBody::Json { value: client_request } = client_request else {
            panic!("fixture request must be JSON")
        };
        assert_eq!(client_request["response_id"], "resp_recorded_0002");
        assert_eq!(client_request["call_id"], "call_recorded_0001");
        assert_eq!(client_request["model"], "resp_model_request");
        assert_eq!(client_request["unrelated_id"], "resp_not_an_identifier");
        let client_response = &fixture.turns[0].client.response.body;
        let RecordedBody::Json { value: client_response } = client_response else {
            panic!("fixture response must be JSON")
        };
        assert_eq!(client_response["id"], "resp_recorded_0002");
        assert_normalized_upstream_ids(fixture);
    }

    fn assert_normalized_upstream_ids(fixture: &WireFixture) {
        let upstream_request = &fixture.turns[0].upstream.request.body;
        let RecordedBody::Json {
            value: upstream_request,
        } = upstream_request
        else {
            panic!("fixture request must be JSON")
        };
        assert_eq!(upstream_request["conversation_id"], "conv_recorded_0004");
        assert_eq!(upstream_request["id"], "resp_recorded_0002");
        assert_eq!(upstream_request["nested"][0]["id"], "msg_recorded_0005");
        assert_eq!(upstream_request["nested"][1]["id"], "chatcmpl-recorded-0006");
        assert_eq!(upstream_request["nested"][2]["id"], "toolu_recorded_0007");
        assert_eq!(upstream_request["nested"][3]["id"], "resp_model_object");
        assert_eq!(upstream_request["nested"][3]["object"], "model");
        let upstream_response = &fixture.turns[0].upstream.response.body;
        let RecordedBody::Json {
            value: upstream_response,
        } = upstream_response
        else {
            panic!("fixture response must be JSON")
        };
        assert_eq!(upstream_response["id"], "resp_recorded_0003");
    }

    #[test]
    fn sanitize_replaces_literals_before_normalizing_and_canonicalizes_dynamic_fields() {
        // Arrange
        let mut fixture = fixture();
        let rules = RedactionRules {
            literals: BTreeMap::from([("literal-source".to_owned(), "resp_literal".to_owned())]),
        };

        // Act
        sanitize_fixture(&mut fixture, &rules).unwrap();

        // Assert
        assert_eq!(fixture.normalization.version, 1);
        let client_request = &fixture.turns[0].client.request.body;
        let RecordedBody::Json { value: client_request } = client_request else {
            panic!("fixture request must be JSON")
        };
        assert_eq!(client_request["arguments"], "{\"a\":{\"x\":3,\"y\":2},\"z\":1}");
        let client_response = &fixture.turns[0].client.response.body;
        let RecordedBody::Json { value: client_response } = client_response else {
            panic!("fixture response must be JSON")
        };
        assert_eq!(client_response["created"], 0);
        assert_eq!(client_response["created_at"], "1970-01-01T00:00:00Z");
        assert!(client_response.get("system_fingerprint").is_none());
        let upstream_response = &fixture.turns[0].upstream.response.body;
        let RecordedBody::Json {
            value: upstream_response,
        } = upstream_response
        else {
            panic!("fixture response must be JSON")
        };
        assert_eq!(upstream_response["previous_response_id"], "resp_recorded_0008");
        assert_eq!(fixture.normalization.linked_ids["resp_literal"], "resp_recorded_0008");
    }

    #[test]
    fn commit_safety_rejects_secrets_paths_headers_and_unredacted_literals_without_echoing_fixture_content() {
        for (value, expected_rule) in [
            (json!("bearer sk-test-secret"), "bearer token"),
            (json!("Basic dXNlcjpwYXNz"), "basic authentication"),
            (json!("/Users/alice/private"), "absolute user path"),
            (json!("/home/alice/private"), "absolute home path"),
            (json!(r"C:\\Users\\alice\\private"), "Windows drive path"),
        ] {
            assert_body_safety_rejected(&value, expected_rule);
        }
        for credential_header in [
            "Authorization",
            "Proxy-Authorization",
            "Cookie",
            "Set-Cookie",
            "X-API-Key",
            "API-Key",
        ] {
            assert_credential_header_rejected(credential_header);
        }
    }

    #[test]
    fn commit_safety_decodes_base64_before_checking_credentials_paths_and_custom_literals() {
        for (decoded, expected_rule) in [
            (b"prefix BeArEr\tcredential".as_slice(), "bearer token"),
            (b"prefix bAsIc\tcredential".as_slice(), "basic authentication"),
            (b"prefix /Users/private/value".as_slice(), "absolute user path"),
            (b"prefix /home/private/value".as_slice(), "absolute home path"),
            (br"prefix C:\private\value".as_slice(), "Windows drive path"),
        ] {
            let mut fixture = commit_safe_fixture();
            fixture.turns[0].client.request.body = base64_body(decoded);

            let message = validate_commit_safe(&fixture).unwrap_err().to_string();

            assert!(message.contains(expected_rule));
            assert!(message.contains("$/turns/0/client/request/body/data"));
            assert!(!message.contains("private"));
            assert!(!message.contains("credential"));
        }

        let rules = RedactionRules {
            literals: BTreeMap::from([("controlled-literal".to_owned(), "[redacted]".to_owned())]),
        };
        let mut fixture = commit_safe_fixture();
        fixture.turns[0].upstream.response.body = base64_body(b"prefix controlled-literal suffix");

        let message = validate_commit_safe_with_rules(&fixture, &rules)
            .unwrap_err()
            .to_string();

        assert!(message.contains("unredacted literal"));
        assert!(message.contains("$/turns/0/upstream/response/body/data"));
        assert!(!message.contains("controlled-literal"));
    }

    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "the table constructs every typed fixture text surface explicitly"
    )]
    fn configured_credentials_are_checked_across_fixture_text_surfaces() {
        const CREDENTIAL: &str = "configured-credential-sentinel";
        let rules = RedactionRules::default();
        let mut configured = HeaderMap::new();
        configured.insert("x-api-key", HeaderValue::from_static(CREDENTIAL));

        let mut provenance = commit_safe_fixture();
        provenance.provenance.model = CREDENTIAL.to_owned();
        let mut normalization = commit_safe_fixture();
        normalization
            .normalization
            .linked_ids
            .insert(CREDENTIAL.to_owned(), "id_recorded_0001".to_owned());
        let mut header = commit_safe_fixture();
        header.turns[0]
            .client
            .request
            .headers
            .insert("request-id".to_owned(), vec![CREDENTIAL.to_owned()]);
        let mut json_key = commit_safe_fixture();
        let mut credential_key = serde_json::Map::new();
        credential_key.insert(CREDENTIAL.to_owned(), json!("safe"));
        json_key.turns[0].client.request.body = RecordedBody::Json {
            value: Value::Object(credential_key),
        };
        let mut sse_metadata = commit_safe_fixture();
        sse_metadata.turns[0].upstream.response.body = RecordedBody::Sse {
            frames: vec![crate::inference_fixture::SseFrame {
                event: Some(CREDENTIAL.to_owned()),
                data: "safe".to_owned(),
                id: None,
                retry: None,
            }],
            done: false,
        };
        let mut base64 = commit_safe_fixture();
        base64.turns[0].client.request.body = base64_body(CREDENTIAL.as_bytes());

        for (case, fixture) in [
            ("provenance", provenance),
            ("normalization", normalization),
            ("header", header),
            ("JSON key", json_key),
            ("SSE metadata", sse_metadata),
            ("decoded Base64", base64),
        ] {
            let error = validate_commit_safe_with_rules_and_credentials(&fixture, &rules, &configured).unwrap_err();
            assert!(
                matches!(
                    error,
                    crate::inference_fixture::FixtureError::CommitSafety {
                        rule: "configured credential",
                        ..
                    }
                ),
                "{case} must reject configured credentials"
            );
            assert!(!error.to_string().contains(CREDENTIAL));
        }
    }

    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "both structured string carriers are constructed and asserted explicitly"
    )]
    fn configured_credentials_are_checked_inside_nested_json_text() {
        const CREDENTIAL: &str = r#"configured-"credential"#;
        let rules = RedactionRules::default();
        let mut configured = HeaderMap::new();
        configured.insert("x-api-key", HeaderValue::from_static(CREDENTIAL));
        let encoded = serde_json::to_string(&json!({"token": CREDENTIAL})).unwrap();

        let mut arguments = commit_safe_fixture();
        arguments.turns[0].client.request.body = RecordedBody::Json {
            value: json!({"arguments": encoded.clone()}),
        };
        let doubly_encoded = serde_json::to_string(&encoded).unwrap();
        let mut nested_arguments = commit_safe_fixture();
        nested_arguments.turns[0].client.request.body = RecordedBody::Json {
            value: json!({"arguments": doubly_encoded.clone()}),
        };
        let mut sse = commit_safe_fixture();
        sse.turns[0].upstream.response.body = RecordedBody::Sse {
            frames: vec![crate::inference_fixture::SseFrame {
                event: None,
                data: serde_json::to_string(&json!({"token": CREDENTIAL})).unwrap(),
                id: None,
                retry: None,
            }],
            done: false,
        };
        let mut credential_key = serde_json::Map::new();
        credential_key.insert(CREDENTIAL.to_owned(), json!("safe"));
        let mut sse_key = commit_safe_fixture();
        sse_key.turns[0].upstream.response.body = RecordedBody::Sse {
            frames: vec![crate::inference_fixture::SseFrame {
                event: None,
                data: serde_json::to_string(&Value::Object(credential_key)).unwrap(),
                id: None,
                retry: None,
            }],
            done: false,
        };
        let mut nested_sse = commit_safe_fixture();
        nested_sse.turns[0].upstream.response.body = RecordedBody::Sse {
            frames: vec![crate::inference_fixture::SseFrame {
                event: None,
                data: doubly_encoded,
                id: None,
                retry: None,
            }],
            done: false,
        };

        for (case, fixture) in [
            ("arguments", arguments),
            ("nested arguments", nested_arguments),
            ("SSE data value", sse),
            ("SSE data key", sse_key),
            ("nested SSE data", nested_sse),
        ] {
            let error = validate_commit_safe_with_rules_and_credentials(&fixture, &rules, &configured).unwrap_err();
            assert!(
                matches!(
                    error,
                    crate::inference_fixture::FixtureError::CommitSafety {
                        rule: "configured credential",
                        ..
                    }
                ),
                "{case} must reject configured credentials in nested JSON"
            );
            assert!(!error.to_string().contains(CREDENTIAL));
        }
    }

    #[test]
    fn structured_json_text_scanning_enforces_depth_and_byte_budgets() {
        let policy = super::CommitSafetyPolicy::default();
        let mut path = super::JsonPath::default();
        let mut deeply_encoded = "null".to_owned();
        for _ in 0..super::MAX_STRUCTURED_JSON_TEXT_DEPTH {
            deeply_encoded = serde_json::to_string(&deeply_encoded).unwrap();
        }
        let mut depth_bytes = usize::MAX;
        let depth_error =
            super::validate_nested_json_text_bounded(&deeply_encoded, policy, &mut path, 0, &mut depth_bytes)
                .unwrap_err();
        let mut insufficient_bytes = deeply_encoded.len() - 1;
        let byte_error =
            super::validate_nested_json_text_bounded(&deeply_encoded, policy, &mut path, 0, &mut insufficient_bytes)
                .unwrap_err();

        assert!(matches!(
            depth_error,
            crate::inference_fixture::FixtureError::CommitSafety {
                rule: "structured JSON text depth limit",
                ..
            }
        ));
        assert!(matches!(
            byte_error,
            crate::inference_fixture::FixtureError::CommitSafety {
                rule: "structured JSON text byte limit",
                ..
            }
        ));
    }

    #[test]
    fn structured_json_text_scanning_fails_closed_before_serde_depth_exhaustion() {
        const CREDENTIAL: &str = r#"configured-"credential"#;
        let rules = RedactionRules::default();
        let mut configured = HeaderMap::new();
        configured.insert("x-api-key", HeaderValue::from_static(CREDENTIAL));
        let encoded_credential = serde_json::to_string(CREDENTIAL).unwrap();
        let deeply_nested = format!("{}{}{}", "[".repeat(129), encoded_credential, "]".repeat(129));
        let mut fixture = commit_safe_fixture();
        fixture.turns[0].client.request.body = RecordedBody::Json {
            value: json!({"arguments": deeply_nested}),
        };

        let error = validate_commit_safe_with_rules_and_credentials(&fixture, &rules, &configured).unwrap_err();

        assert!(matches!(
            error,
            crate::inference_fixture::FixtureError::CommitSafety {
                rule: "structured JSON container depth limit",
                ..
            }
        ));
        assert!(!error.to_string().contains(CREDENTIAL));
    }

    #[test]
    fn sanitize_base64_replaces_decoded_utf8_literals_and_reencodes_the_body() {
        let rules = RedactionRules {
            literals: BTreeMap::from([("controlled-literal".to_owned(), "safe-value".to_owned())]),
        };
        let mut fixture = commit_safe_fixture();
        fixture.turns[0].client.request.body = base64_body(b"before controlled-literal after");

        sanitize_fixture(&mut fixture, &rules).unwrap();

        assert_eq!(decoded_base64_request(&fixture), b"before safe-value after");
        assert!(validate_commit_safe_with_rules(&fixture, &rules).is_ok());
    }

    #[test]
    fn sanitize_base64_is_binary_safe_around_custom_literal_replacements() {
        let rules = RedactionRules {
            literals: BTreeMap::from([("controlled-literal".to_owned(), "safe".to_owned())]),
        };
        let mut fixture = commit_safe_fixture();
        fixture.turns[0].client.request.body = base64_body(b"\xffbefore controlled-literal after\xfe");

        sanitize_fixture(&mut fixture, &rules).unwrap();

        assert_eq!(decoded_base64_request(&fixture), b"\xffbefore safe after\xfe");
        assert!(validate_commit_safe_with_rules(&fixture, &rules).is_ok());
    }

    #[test]
    fn sanitize_base64_binary_context_fails_closed_when_an_unsafe_pattern_remains() {
        let mut fixture = commit_safe_fixture();
        fixture.turns[0].client.request.body = base64_body(b"\xffprefix Bearer\rcredential suffix\xfe");

        let message = sanitize_fixture(&mut fixture, &RedactionRules::default())
            .unwrap_err()
            .to_string();

        assert!(message.contains("bearer token"));
        assert!(message.contains("$/turns/0/client/request/body"));
        assert!(!message.contains("credential"));
    }

    #[test]
    fn base64_sanitization_and_validation_reject_malformed_input_opaquely() {
        let mut fixture = commit_safe_fixture();
        fixture.turns[0].client.request.body = RecordedBody::Base64 {
            data: "%%%controlled-value%%%".to_owned(),
        };

        let sanitize_message = sanitize_fixture(&mut fixture, &RedactionRules::default())
            .unwrap_err()
            .to_string();
        let validate_message = validate_commit_safe(&fixture).unwrap_err().to_string();

        assert_eq!(sanitize_message, "recorded Base64 body is invalid");
        assert_eq!(validate_message, "recorded Base64 body is invalid");
        assert!(!sanitize_message.contains("controlled-value"));
        assert!(!validate_message.contains("controlled-value"));
    }

    #[test]
    fn base64_sanitization_enforces_request_and_response_decoded_limits() {
        use crate::inference_fixture::bounds::{MAX_SCENARIO_REQUEST_BODY_BYTES, MAX_SCRIPTED_RESPONSE_BODY_BYTES};

        let mut fixture = commit_safe_fixture();
        fixture.turns[0].client.request.body = base64_body(&vec![b'x'; MAX_SCENARIO_REQUEST_BODY_BYTES]);
        sanitize_fixture(&mut fixture, &RedactionRules::default()).unwrap();

        fixture.turns[0].client.request.body = base64_body(&vec![b'x'; MAX_SCENARIO_REQUEST_BODY_BYTES + 1]);
        assert_eq!(
            sanitize_fixture(&mut fixture, &RedactionRules::default())
                .unwrap_err()
                .to_string(),
            "scenario request body exceeded replay limit"
        );

        fixture.turns[0].client.request.body = RecordedBody::Empty;
        fixture.turns[0].client.response.body = base64_body(&vec![b'x'; MAX_SCRIPTED_RESPONSE_BODY_BYTES]);
        sanitize_fixture(&mut fixture, &RedactionRules::default()).unwrap();

        fixture.turns[0].client.response.body = base64_body(&vec![b'x'; MAX_SCRIPTED_RESPONSE_BODY_BYTES + 1]);
        assert_eq!(
            sanitize_fixture(&mut fixture, &RedactionRules::default())
                .unwrap_err()
                .to_string(),
            "scripted response body exceeded replay limit"
        );
    }

    fn assert_body_safety_rejected(value: &Value, expected_rule: &str) {
        let mut fixture = fixture();
        sanitize_fixture(&mut fixture, &RedactionRules::default()).unwrap();
        fixture.turns[0].client.request.body = RecordedBody::Json {
            value: json!({"input": value}),
        };
        let message = validate_commit_safe(&fixture).unwrap_err().to_string();
        assert!(message.contains(expected_rule));
        assert!(message.contains('$'));
        assert!(!message.contains("alice"));
        assert!(!message.contains("sk-test-secret"));
    }

    fn assert_credential_header_rejected(credential_header: &str) {
        let mut fixture = fixture();
        sanitize_fixture(&mut fixture, &RedactionRules::default()).unwrap();
        fixture.turns[0].client.request.headers = headers(&[(credential_header, &["must-not-echo"])]);
        let message = validate_commit_safe(&fixture).unwrap_err().to_string();
        assert!(message.contains("credential header"));
        assert!(!message.contains("must-not-echo"));
    }

    #[test]
    fn commit_safety_allows_non_secret_words_that_only_contain_the_literal_marker() {
        // Arrange
        let mut fixture = fixture();
        sanitize_fixture(&mut fixture, &RedactionRules::default()).unwrap();
        fixture.turns[0].client.request.body = RecordedBody::Json {
            value: json!({"input": "secretary"}),
        };

        // Act
        let result = validate_commit_safe(&fixture);

        // Assert
        assert!(result.is_ok());
    }

    #[test]
    fn sanitize_normalizes_sse_json_data_with_the_fixture_wide_identifier_mapping() {
        // Arrange
        let mut fixture = fixture();
        fixture.turns[0].upstream.response.body = RecordedBody::Sse {
            frames: vec![crate::inference_fixture::SseFrame {
                event: Some("response.completed".to_owned()),
                data: "{\"id\":\"resp_source\",\"call_id\":\"call_sse\"}".to_owned(),
                id: Some("resp_source".to_owned()),
                retry: None,
            }],
            done: true,
        };

        // Act
        sanitize_fixture(&mut fixture, &RedactionRules::default()).unwrap();

        // Assert
        let RecordedBody::Sse { frames, .. } = &fixture.turns[0].upstream.response.body else {
            panic!("fixture response must be SSE")
        };
        assert_eq!(
            frames[0].data,
            "{\"call_id\":\"call_recorded_0008\",\"id\":\"resp_recorded_0002\"}"
        );
        assert_eq!(frames[0].id.as_deref(), Some("resp_recorded_0002"));
    }

    #[test]
    fn sanitize_normalizes_non_null_completed_at_and_preserves_null() {
        // Arrange
        let mut fixture = fixture();
        fixture.turns[0].upstream.response.body = RecordedBody::Json {
            value: json!({
                "completed": {"completed_at": 1_786_127_242_u64},
                "in_progress": {"completed_at": null}
            }),
        };

        // Act
        sanitize_fixture(&mut fixture, &RedactionRules::default()).unwrap();

        // Assert
        let RecordedBody::Json { value } = &fixture.turns[0].upstream.response.body else {
            panic!("fixture response must be JSON")
        };
        assert_eq!(value["completed"]["completed_at"], 0);
        assert!(value["in_progress"]["completed_at"].is_null());
    }

    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "the test keeps root and nested obfuscation behavior visible in one audit"
    )]
    fn sanitize_drops_only_top_level_responses_delta_obfuscation_padding() {
        // Arrange
        let mut fixture = fixture();
        fixture.turns[0].upstream.response.body = RecordedBody::Sse {
            frames: vec![
                crate::inference_fixture::SseFrame {
                    event: Some("response.output_text.delta".to_owned()),
                    data: json!({
                        "type": "response.output_text.delta",
                        "delta": "hello",
                        "obfuscation": "provider-padding",
                        "nested": {"obfuscation": "semantic-value"}
                    })
                    .to_string(),
                    id: None,
                    retry: None,
                },
                crate::inference_fixture::SseFrame {
                    event: Some("user.event".to_owned()),
                    data: json!({"type": "user.event", "obfuscation": "user-value"}).to_string(),
                    id: None,
                    retry: None,
                },
            ],
            done: true,
        };

        // Act
        sanitize_fixture(&mut fixture, &RedactionRules::default()).unwrap();

        // Assert
        let RecordedBody::Sse { frames, .. } = &fixture.turns[0].upstream.response.body else {
            panic!("fixture response must be SSE")
        };
        let delta: Value = serde_json::from_str(&frames[0].data).unwrap();
        let user: Value = serde_json::from_str(&frames[1].data).unwrap();
        assert!(delta.get("obfuscation").is_none());
        assert_eq!(delta["nested"]["obfuscation"], "semantic-value");
        assert_eq!(user["obfuscation"], "user-value");
    }

    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "the test keeps the defining item and its cross-event reference visible together"
    )]
    fn sanitize_maps_responses_item_id_to_the_referenced_item_identifier() {
        // Arrange
        let mut fixture = fixture();
        fixture.turns[0].upstream.response.body = RecordedBody::Sse {
            frames: vec![
                crate::inference_fixture::SseFrame {
                    event: Some("response.output_item.added".to_owned()),
                    data: json!({
                        "type": "response.output_item.added",
                        "item": {"id": "msg_source", "type": "message"}
                    })
                    .to_string(),
                    id: None,
                    retry: None,
                },
                crate::inference_fixture::SseFrame {
                    event: Some("response.output_text.delta".to_owned()),
                    data: json!({
                        "type": "response.output_text.delta",
                        "item_id": "msg_source",
                        "delta": "hello"
                    })
                    .to_string(),
                    id: None,
                    retry: None,
                },
            ],
            done: true,
        };

        // Act
        sanitize_fixture(&mut fixture, &RedactionRules::default()).unwrap();

        // Assert
        let RecordedBody::Sse { frames, .. } = &fixture.turns[0].upstream.response.body else {
            panic!("fixture response must be SSE")
        };
        let added: Value = serde_json::from_str(&frames[0].data).unwrap();
        let delta: Value = serde_json::from_str(&frames[1].data).unwrap();
        assert_eq!(delta["item_id"], added["item"]["id"]);
        assert_eq!(delta["item_id"], "msg_recorded_0005");
    }

    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "the fixture keeps root, nested, user, and semantic chunk preservation in one audit"
    )]
    fn sanitize_drops_only_top_level_chat_completion_chunk_obfuscation_padding() {
        // Catches retaining unstable provider padding or globally deleting semantic/user obfuscation fields.
        let mut fixture = fixture();
        fixture.turns[0].client.request.body = RecordedBody::Json {
            value: json!({
                "object": "chat.completion.chunk",
                "obfuscation": "request-user-value",
                "arguments": "{\"object\":\"chat.completion.chunk\",\"obfuscation\":\"user-value\"}"
            }),
        };
        fixture.turns[0].upstream.response.body = RecordedBody::Sse {
            frames: vec![
                crate::inference_fixture::SseFrame {
                    event: None,
                    data: json!({
                        "object": "chat.completion.chunk",
                        "obfuscation": "provider-padding",
                        "choices": [
                            {"index": 0, "delta": {"content": "first"}},
                            {"index": 1, "delta": {"content": "second"}}
                        ],
                        "nested": {
                            "object": "chat.completion.chunk",
                            "obfuscation": "nested-user-value"
                        }
                    })
                    .to_string(),
                    id: None,
                    retry: None,
                },
                crate::inference_fixture::SseFrame {
                    event: None,
                    data: json!({"object": "user.event", "obfuscation": "semantic-value"}).to_string(),
                    id: None,
                    retry: None,
                },
            ],
            done: true,
        };

        sanitize_fixture(&mut fixture, &RedactionRules::default()).unwrap();

        let RecordedBody::Sse { frames, .. } = &fixture.turns[0].upstream.response.body else {
            panic!("fixture response must be SSE")
        };
        let first: Value = serde_json::from_str(&frames[0].data).unwrap();
        let second: Value = serde_json::from_str(&frames[1].data).unwrap();
        assert!(first.get("obfuscation").is_none());
        assert_eq!(first["nested"]["obfuscation"], "nested-user-value");
        assert_eq!(
            first["choices"],
            json!([
                {"delta": {"content": "first"}, "index": 0},
                {"delta": {"content": "second"}, "index": 1}
            ])
        );
        assert_eq!(second["obfuscation"], "semantic-value");
        let RecordedBody::Json { value } = &fixture.turns[0].client.request.body else {
            panic!("fixture request must be JSON")
        };
        assert_eq!(value["obfuscation"], "request-user-value");
        let arguments: Value = serde_json::from_str(value["arguments"].as_str().unwrap()).unwrap();
        assert_eq!(arguments["object"], "chat.completion.chunk");
        assert_eq!(arguments["obfuscation"], "user-value");
    }

    #[test]
    fn commit_safety_with_rules_rejects_arbitrary_literals_without_rejecting_legitimate_secret_content() {
        // Arrange
        let rules = RedactionRules {
            literals: BTreeMap::from([("customer-acme-42".to_owned(), "[redacted]".to_owned())]),
        };
        let mut fixture = fixture();
        sanitize_fixture(&mut fixture, &RedactionRules::default()).unwrap();
        fixture.turns[0].client.request.body = RecordedBody::Json {
            value: json!({"nested": {"input": "customer-acme-42"}}),
        };

        // Act
        let base_result = validate_commit_safe(&fixture);
        let error = validate_commit_safe_with_rules(&fixture, &rules).unwrap_err();
        let message = error.to_string();
        fixture.turns[0].client.request.body = RecordedBody::Json {
            value: json!({"input": "a legitimate secret"}),
        };
        let secret_result = validate_commit_safe_with_rules(&fixture, &rules);

        // Assert
        assert!(base_result.is_ok());
        assert!(message.contains("unredacted literal"));
        assert!(message.contains("$/turns/0/client/request/body/value/<key>/<key>"));
        assert!(!message.contains("customer-acme-42"));
        assert!(secret_result.is_ok());
    }

    #[test]
    fn sanitize_replaces_literals_once_without_cascading_across_json_sse_and_arguments() {
        // Arrange
        let rules = RedactionRules {
            literals: BTreeMap::from([("a".to_owned(), "x".to_owned()), ("b".to_owned(), "a".to_owned())]),
        };
        let mut recording = fixture();
        recording.turns[0].client.request.body = RecordedBody::Json {
            value: json!({"input": "b", "arguments": "{\"input\":\"b\"}"}),
        };
        recording.turns[0].upstream.response.body = RecordedBody::Sse {
            frames: vec![crate::inference_fixture::SseFrame {
                event: None,
                data: "{\"input\":\"b\"}".to_owned(),
                id: None,
                retry: None,
            }],
            done: true,
        };

        // Act
        sanitize_fixture(&mut recording, &rules).unwrap();

        // Assert
        let RecordedBody::Json { value } = &recording.turns[0].client.request.body else {
            panic!("fixture request must be JSON")
        };
        assert_eq!(value["input"], "a");
        assert_eq!(value["arguments"], "{\"input\":\"a\"}");
        let RecordedBody::Sse { frames, .. } = &recording.turns[0].upstream.response.body else {
            panic!("fixture response must be SSE")
        };
        assert_eq!(frames[0].data, "{\"input\":\"a\"}");
    }

    #[test]
    fn sanitize_prefers_longest_overlapping_literal_and_rejects_empty_sources() {
        // Arrange
        let mut recording = fixture();
        recording.turns[0].client.request.body = RecordedBody::Json {
            value: json!({"input": "customer-acme-42"}),
        };
        let overlapping_rules = RedactionRules {
            literals: BTreeMap::from([
                ("customer-acme".to_owned(), "short".to_owned()),
                ("customer-acme-42".to_owned(), "long".to_owned()),
            ]),
        };
        let empty_source_rules = RedactionRules {
            literals: BTreeMap::from([(String::new(), "invalid".to_owned())]),
        };

        // Act
        sanitize_fixture(&mut recording, &overlapping_rules).unwrap();
        let mut invalid_fixture = fixture();
        let empty_result = sanitize_fixture(&mut invalid_fixture, &empty_source_rules);

        // Assert
        let RecordedBody::Json { value } = &recording.turns[0].client.request.body else {
            panic!("fixture request must be JSON")
        };
        assert_eq!(value["input"], "long");
        assert!(empty_result.is_err());
    }

    #[test]
    fn commit_safety_distinguishes_absolute_paths_from_urls_and_reports_nested_paths() {
        for (path, rule) in [
            ("C:/Users/alice/private", "Windows drive path"),
            ("c:/Users/alice/private", "Windows drive path"),
            (r"C:\Users\alice\private", "Windows drive path"),
            ("/Users/alice/private", "absolute user path"),
            ("/home/alice/private", "absolute home path"),
        ] {
            let mut fixture = fixture();
            sanitize_fixture(&mut fixture, &RedactionRules::default()).unwrap();
            fixture.turns[0].client.request.body = RecordedBody::Json {
                value: json!({"nested": {"input": path}}),
            };
            let message = validate_commit_safe(&fixture).unwrap_err().to_string();
            assert!(message.contains(rule));
            assert!(message.contains("$/turns/0/client/request/body/value/<key>/<key>"));
            assert!(!message.contains("alice"));
        }
        let mut fixture = fixture();
        sanitize_fixture(&mut fixture, &RedactionRules::default()).unwrap();
        fixture.turns[0].client.request.body = RecordedBody::Json {
            value: json!({
                "url": "https://host/home/docs",
                "prose": "home/docs is relative",
                "user_boundary": "/Usersx/alice",
                "home_boundary": "/homex/alice",
            }),
        };
        assert!(validate_commit_safe(&fixture).is_ok());
    }

    #[test]
    fn commit_safety_matches_embedded_paths_and_all_ascii_bearer_whitespace_at_lexical_boundaries() {
        for (text, expected_rule) in [
            ("prefix=/Users/private/value", "absolute user path"),
            ("prefix: /home/private/value", "absolute home path"),
            (r"prefix=(C:\private\value)", "Windows drive path"),
            ("prefix Bearer\tcredential", "bearer token"),
            ("prefix bearer\ncredential", "bearer token"),
            ("prefix BEARER\u{000b}credential", "bearer token"),
            ("prefix Bearer\u{000c}credential", "bearer token"),
            ("prefix Bearer\rcredential", "bearer token"),
        ] {
            let mut fixture = commit_safe_fixture();
            fixture.turns[0].client.request.body = RecordedBody::Json {
                value: json!({"unknown": text}),
            };

            let message = validate_commit_safe(&fixture).unwrap_err().to_string();

            assert!(message.contains(expected_rule));
            assert!(!message.contains("private"));
            assert!(!message.contains("credential"));
        }
    }

    #[test]
    fn commit_safety_allows_http_url_paths_but_rejects_local_file_urls() {
        let mut fixture = commit_safe_fixture();
        fixture.turns[0].client.request.body = RecordedBody::Json {
            value: json!({
                "ipv4": "https://example.test/home/private/value",
                "ipv6": "http://[::1]/Users/private/value",
                "drive_component": "https://example.test/C:/private/value"
            }),
        };
        assert!(validate_commit_safe(&fixture).is_ok());

        fixture.turns[0].client.request.body = RecordedBody::Json {
            value: json!({"unknown": "file:///home/private/value"}),
        };
        let message = validate_commit_safe(&fixture).unwrap_err().to_string();
        assert!(message.contains("absolute home path"));
        assert!(!message.contains("private"));

        for value in [
            "https://example.test/docs?next=/home/private/value",
            "https://example.test/docs#/Users/private/value",
        ] {
            fixture.turns[0].client.request.body = RecordedBody::Json {
                value: json!({"unknown": value}),
            };
            let message = validate_commit_safe(&fixture).unwrap_err().to_string();
            assert!(message.contains("absolute"));
            assert!(!message.contains("private"));
        }
    }

    #[test]
    fn commit_safety_allows_application_json_objects_named_headers() {
        // Arrange
        let mut fixture = fixture();
        sanitize_fixture(&mut fixture, &RedactionRules::default()).unwrap();
        fixture.turns[0].client.request.body = RecordedBody::Json {
            value: json!({"headers": {"Authorization": "application metadata"}}),
        };

        // Act
        let result = validate_commit_safe(&fixture);

        // Assert
        assert!(result.is_ok());
    }

    /// Proves every JSON object key is checked with the same safety rules as values.
    #[test]
    fn commit_safety_rejects_unsafe_json_object_keys() {
        let literal_rules = RedactionRules {
            literals: BTreeMap::from([("customer-acme-42".to_owned(), "[redacted]".to_owned())]),
        };
        let literal_message = commit_safety_message_for_key("customer-acme-42", Some(&literal_rules));
        assert!(literal_message.contains("unredacted literal"));
        assert!(!literal_message.contains("customer-acme-42"));

        for (key, expected_rule) in [
            ("Bearer key-token", "bearer token"),
            ("/Users/alice/private", "absolute user path"),
            ("/home/alice/private", "absolute home path"),
            (r"C:\\Users\\alice\\private", "Windows drive path"),
            ("C:/Users/alice/private", "Windows drive path"),
        ] {
            let message = commit_safety_message_for_key(key, None);
            assert!(message.contains(expected_rule));
            assert!(!message.contains(key));
        }
    }

    /// Proves unsafe descendant diagnostics do not reveal arbitrary body object keys.
    #[test]
    fn commit_safety_hides_sensitive_body_keys_from_descendant_paths() {
        let sensitive_key = "customer-acme-42";
        let mut fixture = commit_safe_fixture();
        fixture.turns[0].client.request.body = RecordedBody::Json {
            value: json!({sensitive_key: {"input": "Bearer child-token"}}),
        };

        let message = validate_commit_safe(&fixture).unwrap_err().to_string();

        assert!(message.contains("bearer token"));
        assert!(message.contains("$/turns/0/client/request/body/value/<key>/<key>"));
        assert!(!message.contains(sensitive_key));
        assert!(!message.contains("child-token"));
    }

    /// Proves nested application data cannot masquerade as a wire header map.
    #[test]
    fn commit_safety_treats_nested_header_suffixes_as_application_json() {
        let mut fixture = commit_safe_fixture();
        fixture.turns[0].client.request.body = RecordedBody::Json {
            value: json!({
                "turns": [{
                    "client": {
                        "request": {
                            "headers": {"Authorization": "application metadata"}
                        }
                    }
                }]
            }),
        };

        assert!(validate_commit_safe(&fixture).is_ok());
    }

    /// Proves credential-name validation still applies to a real wire header map.
    #[test]
    fn commit_safety_rejects_credential_names_in_real_wire_header_maps() {
        let mut fixture = commit_safe_fixture();
        fixture.turns[0].client.request.headers = headers(&[("Authorization", &["discarded"])]);

        let message = validate_commit_safe(&fixture).unwrap_err().to_string();

        assert!(message.contains("credential header"));
        assert!(message.contains("$/turns/0/client/request/headers/<key>"));
        assert!(!message.contains("discarded"));
    }

    #[test]
    fn sanitize_uses_lexical_object_key_order_for_identifier_sequences() {
        // Arrange
        let first = object_with_identifier_order([("id", "resp_source"), ("call_id", "call_source")]);
        let second = object_with_identifier_order([("call_id", "call_source"), ("id", "resp_source")]);
        let mut first_fixture = fixture_with_single_json_body(first);
        let mut second_fixture = fixture_with_single_json_body(second);

        // Act
        sanitize_fixture(&mut first_fixture, &RedactionRules::default()).unwrap();
        sanitize_fixture(&mut second_fixture, &RedactionRules::default()).unwrap();

        // Assert
        assert_eq!(
            first_fixture.normalization.linked_ids,
            second_fixture.normalization.linked_ids
        );
        assert_eq!(
            first_fixture.turns[0].client.request.body,
            second_fixture.turns[0].client.request.body
        );
        assert_eq!(
            first_fixture.normalization.linked_ids,
            BTreeMap::from([
                ("call_source".to_owned(), "call_recorded_0001".to_owned()),
                ("resp_source".to_owned(), "resp_recorded_0002".to_owned()),
            ])
        );
    }

    fn object_with_identifier_order(entries: [(&str, &str); 2]) -> Value {
        let mut object = serde_json::Map::new();
        for (key, value) in entries {
            object.insert(key.to_owned(), Value::String(value.to_owned()));
        }
        Value::Object(object)
    }

    fn fixture_with_single_json_body(value: Value) -> WireFixture {
        let mut fixture = fixture();
        fixture.turns[0].client.request.body = RecordedBody::Json { value };
        fixture.turns[0].client.response.body = RecordedBody::Empty;
        fixture.turns[0].upstream.request.body = RecordedBody::Empty;
        fixture.turns[0].upstream.response.body = RecordedBody::Empty;
        fixture
    }

    /// Returns a baseline fixture after its recorded dynamic data has been normalized.
    fn commit_safe_fixture() -> WireFixture {
        let mut fixture = fixture();
        sanitize_fixture(&mut fixture, &RedactionRules::default()).unwrap();
        fixture
    }

    /// Returns the rule-safe validation diagnostic for one unsafe JSON object key.
    fn commit_safety_message_for_key(key: &str, rules: Option<&RedactionRules>) -> String {
        let mut fixture = commit_safe_fixture();
        let mut body = serde_json::Map::new();
        body.insert(key.to_owned(), json!("safe value"));
        fixture.turns[0].client.request.body = RecordedBody::Json {
            value: Value::Object(body),
        };
        match rules {
            Some(rules) => validate_commit_safe_with_rules(&fixture, rules)
                .unwrap_err()
                .to_string(),
            None => validate_commit_safe(&fixture).unwrap_err().to_string(),
        }
    }

    fn base64_body(decoded: &[u8]) -> RecordedBody {
        RecordedBody::Base64 {
            data: STANDARD.encode(decoded),
        }
    }

    fn decoded_base64_request(fixture: &WireFixture) -> Vec<u8> {
        let RecordedBody::Base64 { data } = &fixture.turns[0].client.request.body else {
            panic!("fixture request must be Base64")
        };
        STANDARD.decode(data).unwrap()
    }
}
