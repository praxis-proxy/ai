// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Provider-neutral runtime operation identity.
//!
//! One AI application protocol is not one provider, and one provider is not one
//! protocol: OpenAI serves Responses, Conversations, and Chat Completions, while
//! Anthropic Messages is a protocol of its own. This module models the identity
//! a request head resolves to — protocol, operation, method, transport, path
//! template, and request-body shape — without naming any provider.
//!
//! Contract generation stays with the provider that owns it. A registry may
//! wrap the shared `OperationSpec` in a richer type carrying provider-specific
//! metadata and expose the shared part through `OperationEntry`, so the matcher
//! reads one representation while `OpenAPI` generation keeps its own.

/// Application protocol an operation belongs to.
///
/// Deliberately open-ended rather than an enum: a registry declares its own
/// identifier beside its operations, so registering a protocol does not edit a
/// central list every provider must agree on. Values are provider-qualified so
/// `openai_responses` and `anthropic_messages` share one identifier space.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ApplicationProtocol(&'static str);

impl ApplicationProtocol {
    /// Declare one protocol identifier.
    ///
    /// Each registry declares its own beside its operations, so adding a
    /// protocol does not touch this module.
    ///
    /// # Panics
    ///
    /// Panics when the identifier is empty. Identifiers are `'static` and
    /// declared as constants, so this is a compile-time error at every real
    /// call site rather than a runtime failure.
    #[must_use]
    pub const fn new(value: &'static str) -> Self {
        assert!(!value.is_empty(), "application protocol identifier must not be empty");
        Self(value)
    }

    /// Stable label published to downstream filters and routing headers.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        self.0
    }
}

/// HTTP method recognized by an operation registry.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum HttpMethod {
    /// `GET`.
    Get,
    /// `POST`.
    Post,
    /// `PUT`.
    Put,
    /// `DELETE`.
    Delete,
    /// `PATCH`.
    Patch,
    /// `HEAD`.
    Head,
    /// `OPTIONS`.
    Options,
    /// `TRACE`.
    Trace,
}

impl HttpMethod {
    /// Stable uppercase spelling used by runtime matching and reports.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Get => "GET",
            Self::Post => "POST",
            Self::Put => "PUT",
            Self::Delete => "DELETE",
            Self::Patch => "PATCH",
            Self::Head => "HEAD",
            Self::Options => "OPTIONS",
            Self::Trace => "TRACE",
        }
    }
}

/// Transport an operation is reached over.
///
/// Method and path alone cannot separate a plain `GET` from a `WebSocket`
/// handshake at the same address, so transport is part of operation identity.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Transport {
    /// Ordinary HTTP request.
    Http,
    /// `WebSocket` upgrade.
    WebSocket,
}

impl Transport {
    /// Stable label used by reports and tracing.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Http => "http",
            Self::WebSocket => "websocket",
        }
    }
}

/// How the proxy handles an operation at its boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HandlingMode {
    /// Forward the operation without inspecting its payload.
    Passthrough,
    /// Read selected fields while preserving the forwarded payload.
    Inspect,
    /// Rewrite the operation between input and output contracts.
    Transform,
    /// Terminate the request and produce the response locally.
    Local,
}

impl HandlingMode {
    /// Stable label used by conformance reports.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Passthrough => "passthrough",
            Self::Inspect => "inspect",
            Self::Transform => "transform",
            Self::Local => "local",
        }
    }
}

/// Runtime request-body shape, independent of contract ownership.
///
/// The proxy still needs the body shape of operations it merely proxies: a
/// multipart file upload has a body whether or not Praxis describes that body
/// in a generated `OpenAPI` document.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RequestBody {
    /// Operation takes no request body.
    None,
    /// `application/json` body.
    Json {
        /// Whether the body must be present.
        required: bool,
    },
    /// `multipart/form-data` body.
    Multipart {
        /// Whether the body must be present.
        required: bool,
    },
    /// Opaque binary body.
    Binary {
        /// Whether the body must be present.
        required: bool,
    },
}

impl RequestBody {
    /// Whether the operation carries a request body at all.
    #[must_use]
    pub const fn is_present(self) -> bool {
        !matches!(self, Self::None)
    }

    /// Whether a request body must be supplied by the client.
    #[must_use]
    pub const fn is_required(self) -> bool {
        match self {
            Self::None => false,
            Self::Json { required } | Self::Multipart { required } | Self::Binary { required } => required,
        }
    }

    /// Stable label used by reports and tracing.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Json { .. } => "json",
            Self::Multipart { .. } => "multipart",
            Self::Binary { .. } => "binary",
        }
    }
}

/// Runtime identity of one operation, shared by every registry.
///
/// Carries only what resolving a request head requires. Provider-owned
/// specification provenance and generated contracts live in the provider's own
/// wrapper type.
#[derive(Clone, Copy, Debug)]
pub(crate) struct OperationSpec {
    /// Application protocol that owns the operation.
    pub application_protocol: ApplicationProtocol,
    /// Stable operation ID.
    pub operation_id: &'static str,
    /// Typed HTTP method.
    pub method: HttpMethod,
    /// Transport the operation is reached over.
    pub transport: Transport,
    /// Runtime path template handled by Praxis.
    pub runtime_path: &'static str,
    /// Proxy handling mode.
    pub mode: HandlingMode,
    /// Runtime request-body shape.
    pub request_body: RequestBody,
}

impl OperationSpec {
    /// Whether this operation consumes a request body.
    ///
    /// Answers the runtime question directly rather than inferring it from
    /// contract ownership, so proxied operations report their real body shape.
    #[must_use]
    #[cfg(all(test, feature = "openai-conversations"))]
    pub(crate) const fn has_request_body(&self) -> bool {
        self.request_body.is_present()
    }

    /// Number of literal segments in the runtime path template.
    ///
    /// Used to rank candidates so a literal segment always outranks a
    /// parameter, independent of the order operations were declared in.
    fn static_segment_count(&self) -> usize {
        self.runtime_path
            .split('/')
            .filter(|segment| !is_parameter_segment(segment))
            .count()
    }
}

/// One entry in an operation registry.
///
/// Registries may hold bare [`OperationSpec`] values or wrap them in a richer
/// type carrying provider-specific data, so the shared matcher reads the shared
/// metadata through this trait rather than requiring one representation.
pub(crate) trait OperationEntry {
    /// Borrow the shared operation metadata for this entry.
    fn spec(&self) -> &OperationSpec;
}

impl OperationEntry for OperationSpec {
    fn spec(&self) -> &Self {
        self
    }
}

/// Maximum path parameters captured from any one operation template.
///
/// The deepest OpenAI path templates carry two parameters. The headroom keeps
/// the capacity from binding as other protocols register their operations, and
/// a registry test fails if a template ever exceeds it.
pub(crate) const MAX_PATH_PARAMS: usize = 4;

/// Path parameters borrowed directly from the request URI.
///
/// Parameters are held as name/value pairs rather than named fields so that one
/// matcher can serve every protocol. The fixed-capacity array keeps matching
/// allocation-free on the request path.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct RouteParams<'a> {
    /// Captured parameter names paired with their borrowed path segments.
    pairs: [(&'static str, &'a str); MAX_PATH_PARAMS],
    /// Number of occupied slots.
    len: usize,
}

impl<'a> RouteParams<'a> {
    /// Record one captured parameter, failing when capacity is exhausted.
    fn insert(&mut self, name: &'static str, value: &'a str) -> Option<()> {
        // A template repeating a name, such as `{id}/{id}`, has no single
        // answer for `get`, so the match fails rather than silently keeping
        // the first segment.
        if self.get(name).is_some() {
            return None;
        }
        let slot = self.pairs.get_mut(self.len)?;
        *slot = (name, value);
        self.len += 1;
        Some(())
    }

    /// Borrow one captured parameter by template name.
    #[must_use]
    pub(crate) fn get(&self, name: &str) -> Option<&'a str> {
        self.pairs
            .get(..self.len)?
            .iter()
            .find(|(candidate, _)| *candidate == name)
            .map(|&(_, value)| value)
    }

    /// Convert borrowed values into checked byte offsets in their source path.
    ///
    /// Request extensions require `'static` values, so downstream filters
    /// cannot retain these borrows. Offsets preserve allocation-free access to
    /// the immutable request path without extending the borrow's lifetime.
    pub(crate) fn offsets_in(&self, path: &str) -> Option<PathParameterOffsets> {
        let path_start = path.as_ptr() as usize;
        let path_end = path_start.checked_add(path.len())?;
        let mut offsets = PathParameterOffsets::default();

        for &(name, value) in self.pairs.get(..self.len)? {
            let start_address = value.as_ptr() as usize;
            let end_address = start_address.checked_add(value.len())?;
            if start_address < path_start || end_address > path_end {
                return None;
            }
            let start = start_address.checked_sub(path_start)?;
            let end = start.checked_add(value.len())?;
            if path.get(start..end) != Some(value) {
                return None;
            }
            offsets.insert(name, start, end)?;
        }

        Some(offsets)
    }
}

/// Allocation-free path-parameter locations in an immutable request path.
///
/// The matcher validates every byte range when constructing this value. The
/// caller must recover parameters from that same immutable path. Consumers
/// still use [`str::get`], so incompatible bounds or UTF-8 boundaries fail
/// closed instead of panicking.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PathParameterOffsets {
    /// Parameter names paired with start/end byte offsets.
    pairs: [(&'static str, usize, usize); MAX_PATH_PARAMS],
    /// Number of occupied slots.
    len: usize,
}

impl Default for PathParameterOffsets {
    fn default() -> Self {
        Self {
            pairs: [("", 0, 0); MAX_PATH_PARAMS],
            len: 0,
        }
    }
}

impl PathParameterOffsets {
    /// Record one validated byte range.
    fn insert(&mut self, name: &'static str, start: usize, end: usize) -> Option<()> {
        if self
            .pairs
            .get(..self.len)?
            .iter()
            .any(|(candidate, ..)| *candidate == name)
        {
            return None;
        }
        let slot = self.pairs.get_mut(self.len)?;
        *slot = (name, start, end);
        self.len += 1;
        Some(())
    }

    /// Recover one parameter by name from the current immutable request path.
    #[must_use]
    pub fn get<'a>(&self, path: &'a str, name: &str) -> Option<&'a str> {
        let &(_, start, end) = self
            .pairs
            .get(..self.len)?
            .iter()
            .find(|(candidate, ..)| *candidate == name)?;
        path.get(start..end)
    }

    /// Number of captured path parameters.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Whether the matched path captured no parameters.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// One operation matched from a request head.
#[derive(Clone, Copy)]
pub(crate) struct MatchedOperation<'a, T: 'static> {
    /// Matched registry entry, in the registry's own wrapper type.
    pub(crate) spec: &'static T,
    /// Borrowed path parameters.
    pub(crate) params: RouteParams<'a>,
}

/// Whether one template segment declares a path parameter.
fn is_parameter_segment(segment: &str) -> bool {
    segment.starts_with('{') && segment.ends_with('}')
}

/// Normalize a request path before matching.
///
/// Applies the shared policy: any query string is ignored, and exactly one
/// trailing slash is tolerated on a non-root path. Callers that already pass a
/// query-free path are unaffected.
fn normalize_path(path: &str) -> &str {
    let path = path.split('?').next().unwrap_or(path);
    path.strip_suffix('/').filter(|path| !path.is_empty()).unwrap_or(path)
}

/// Match a request head against one operation registry.
///
/// Methods and transports must match exactly. Among templates that match the
/// path, the one with the most literal segments wins, so a static endpoint is
/// never consumed as a path parameter regardless of declaration order.
///
/// Nothing from the request is cloned: path parameters borrow directly from the
/// caller's path.
pub(crate) fn match_operation<'a, T>(
    specs: &'static [T],
    method: &str,
    path: &'a str,
    transport: Transport,
) -> Option<MatchedOperation<'a, T>>
where
    T: OperationEntry,
{
    let path = normalize_path(path);
    specs
        .iter()
        .filter(|entry| {
            let spec = entry.spec();
            spec.method.as_str() == method && spec.transport == transport
        })
        .filter_map(|entry| match_path_template(entry.spec().runtime_path, path).map(|params| (entry, params)))
        .max_by_key(|(entry, _)| entry.spec().static_segment_count())
        .map(|(spec, params)| MatchedOperation { spec, params })
}

/// Match one path against a template with `{param}` placeholders.
///
/// Each parameter matches exactly one non-empty segment.
fn match_path_template<'a>(template: &'static str, path: &'a str) -> Option<RouteParams<'a>> {
    let mut template_segments = template.split('/');
    let mut path_segments = path.split('/');
    let mut params = RouteParams::default();

    loop {
        match (template_segments.next(), path_segments.next()) {
            (None, None) => return Some(params),
            (Some(template_segment), Some(path_segment)) => {
                if let Some(name) = template_segment
                    .strip_prefix('{')
                    .and_then(|segment| segment.strip_suffix('}'))
                {
                    if path_segment.is_empty() {
                        return None;
                    }
                    params.insert(name, path_segment)?;
                } else if template_segment != path_segment {
                    return None;
                }
            },
            _ => return None,
        }
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "tests")]
mod tests {
    use super::*;

    /// Protocol identifiers for this module's tests only. Real registries
    /// declare their own beside their operations.
    const OPENAI_RESPONSES: ApplicationProtocol = ApplicationProtocol::new("openai_responses");
    /// Stand-in for a second protocol, to show the matcher does not branch.
    const ANTHROPIC_MESSAGES: ApplicationProtocol = ApplicationProtocol::new("anthropic_messages");

    /// Two protocols sharing a matcher, declared without touching this module.
    const SPECS: &[OperationSpec] = &[
        OperationSpec {
            application_protocol: OPENAI_RESPONSES,
            operation_id: "createResponse",
            method: HttpMethod::Post,
            transport: Transport::Http,
            runtime_path: "/v1/responses",
            mode: HandlingMode::Transform,
            request_body: RequestBody::Json { required: true },
        },
        OperationSpec {
            application_protocol: OPENAI_RESPONSES,
            operation_id: "getResponse",
            method: HttpMethod::Get,
            transport: Transport::Http,
            runtime_path: "/v1/responses/{response_id}",
            mode: HandlingMode::Transform,
            request_body: RequestBody::None,
        },
        OperationSpec {
            application_protocol: OPENAI_RESPONSES,
            operation_id: "getInputTokenCounts",
            method: HttpMethod::Get,
            transport: Transport::Http,
            runtime_path: "/v1/responses/input_tokens",
            mode: HandlingMode::Local,
            request_body: RequestBody::None,
        },
        OperationSpec {
            application_protocol: ANTHROPIC_MESSAGES,
            operation_id: "createMessage",
            method: HttpMethod::Post,
            transport: Transport::Http,
            runtime_path: "/v1/messages",
            mode: HandlingMode::Passthrough,
            request_body: RequestBody::Json { required: true },
        },
    ];

    /// The slice below is hand-built. It shows the matcher does not branch on
    /// protocol; it is not evidence that two shipped registries share it. Only
    /// OpenAI registries exist today.
    #[test]
    fn the_matcher_does_not_branch_on_protocol() {
        let responses = match_operation(SPECS, "POST", "/v1/responses", Transport::Http).unwrap();
        assert_eq!(responses.spec.application_protocol, OPENAI_RESPONSES);

        let messages = match_operation(SPECS, "POST", "/v1/messages", Transport::Http).unwrap();
        assert_eq!(
            messages.spec.application_protocol, ANTHROPIC_MESSAGES,
            "the matcher must not branch on provider"
        );
    }

    #[test]
    fn a_literal_segment_outranks_a_parameter() {
        let matched = match_operation(SPECS, "GET", "/v1/responses/input_tokens", Transport::Http).unwrap();
        assert_eq!(
            matched.spec.operation_id, "getInputTokenCounts",
            "a static endpoint must not be consumed as an identifier"
        );
    }

    #[test]
    fn parameters_borrow_from_the_request_path() {
        let path = "/v1/responses/resp_123";
        let matched = match_operation(SPECS, "GET", path, Transport::Http).unwrap();
        assert_eq!(matched.spec.operation_id, "getResponse");
        assert_eq!(matched.params.get("response_id"), Some("resp_123"));
        assert_eq!(matched.params.get("missing"), None);

        let offsets = matched.params.offsets_in(path).unwrap();
        assert_eq!(offsets.get(path, "response_id"), Some("resp_123"));
        assert_eq!(offsets.get(path, "missing"), None);
        assert_eq!(offsets.len(), 1);
        assert!(!offsets.is_empty());
        assert_eq!(
            offsets.get("/changed", "response_id"),
            None,
            "offset recovery must fail closed when the source path is unavailable"
        );
    }

    #[test]
    fn rejects_duplicate_parameter_name() {
        let mut params = RouteParams::default();
        assert!(params.insert("id", "first").is_some());
        assert!(
            params.insert("id", "second").is_none(),
            "a repeated parameter name must fail the match rather than overwrite"
        );
        assert_eq!(params.get("id"), Some("first"));
    }

    #[test]
    fn rejects_more_parameters_than_capacity() {
        let names = ["a", "b", "c", "d"];
        assert_eq!(names.len(), MAX_PATH_PARAMS, "this test must fill exactly the capacity");

        let mut params = RouteParams::default();
        for name in names {
            assert!(params.insert(name, "value").is_some());
        }
        assert!(
            params.insert("overflow", "value").is_none(),
            "exceeding capacity must fail the match rather than drop a parameter"
        );
    }

    #[test]
    fn a_template_repeating_a_parameter_name_does_not_match() {
        const REPEATED: &[OperationSpec] = &[OperationSpec {
            application_protocol: ApplicationProtocol::new("test_protocol"),
            operation_id: "repeated",
            method: HttpMethod::Get,
            transport: Transport::Http,
            runtime_path: "/v1/thing/{id}/{id}",
            mode: HandlingMode::Passthrough,
            request_body: RequestBody::None,
        }];

        assert!(
            match_operation(REPEATED, "GET", "/v1/thing/a/b", Transport::Http).is_none(),
            "a template cannot bind one name twice"
        );
    }

    #[test]
    fn transport_is_part_of_identity() {
        assert!(match_operation(SPECS, "POST", "/v1/responses", Transport::WebSocket).is_none());
    }

    #[test]
    fn query_strings_and_one_trailing_slash_are_tolerated() {
        assert!(match_operation(SPECS, "POST", "/v1/responses?stream=true", Transport::Http).is_some());
        assert!(match_operation(SPECS, "POST", "/v1/responses/", Transport::Http).is_some());
        assert!(match_operation(SPECS, "POST", "/v1/responses//", Transport::Http).is_none());
    }

    #[test]
    #[should_panic(expected = "must not be empty")]
    fn an_empty_protocol_identifier_is_rejected() {
        let _empty = ApplicationProtocol::new("");
    }

    #[test]
    fn protocol_identifiers_are_open_ended() {
        const CUSTOM: ApplicationProtocol = ApplicationProtocol::new("vendor_custom_api");
        assert_eq!(CUSTOM.as_str(), "vendor_custom_api");
        assert_eq!(OPENAI_RESPONSES.as_str(), "openai_responses");
        assert_eq!(ANTHROPIC_MESSAGES.as_str(), "anthropic_messages");
    }
}
