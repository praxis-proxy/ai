// SPDX-License-Identifier: MIT
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
//! wrap [`OperationSpec`] in a richer type carrying provider-specific metadata
//! and expose the shared part through [`OperationEntry`], so the matcher reads
//! one representation while `OpenAPI` generation keeps its own.

/// Application protocol an operation belongs to.
///
/// Deliberately open-ended rather than an enum: registering a new protocol is
/// adding a constant beside its registry, not editing a central list every
/// provider must agree on. Values are provider-qualified so `openai_responses`
/// and `anthropic_messages` share one identifier space.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ApplicationProtocol(&'static str);

impl ApplicationProtocol {
    /// Anthropic Messages API.
    pub const ANTHROPIC_MESSAGES: Self = Self("anthropic_messages");
    /// `OpenAI` Chat Completions API.
    pub const OPENAI_CHAT_COMPLETIONS: Self = Self("openai_chat_completions");
    /// `OpenAI` Conversations API.
    pub const OPENAI_CONVERSATIONS: Self = Self("openai_conversations");
    /// `OpenAI` Files API.
    pub const OPENAI_FILES: Self = Self("openai_files");
    /// `OpenAI` Responses API.
    pub const OPENAI_RESPONSES: Self = Self("openai_responses");
    /// `OpenAI` Vector Stores API.
    pub const OPENAI_VECTOR_STORES: Self = Self("openai_vector_stores");

    /// Declare a protocol identifier not covered by the canonical constants.
    #[must_use]
    pub const fn new(value: &'static str) -> Self {
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

    /// Whether Praxis owns the operation's externally visible contract.
    #[must_use]
    pub const fn owns_contract(self) -> bool {
        matches!(self, Self::Transform | Self::Local)
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
pub struct OperationSpec {
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
    pub const fn has_request_body(&self) -> bool {
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
pub trait OperationEntry {
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
pub const MAX_PATH_PARAMS: usize = 4;

/// Path parameters borrowed directly from the request URI.
///
/// Parameters are held as name/value pairs rather than named fields so that one
/// matcher can serve every protocol. The fixed-capacity array keeps matching
/// allocation-free on the request path.
#[derive(Clone, Copy, Debug, Default)]
pub struct RouteParams<'a> {
    /// Captured parameter names paired with their borrowed path segments.
    pairs: [(&'static str, &'a str); MAX_PATH_PARAMS],
    /// Number of occupied slots.
    len: usize,
}

impl<'a> RouteParams<'a> {
    /// Record one captured parameter, failing when capacity is exhausted.
    fn insert(&mut self, name: &'static str, value: &'a str) -> Option<()> {
        let slot = self.pairs.get_mut(self.len)?;
        *slot = (name, value);
        self.len += 1;
        Some(())
    }

    /// Borrow one captured parameter by template name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&'a str> {
        self.pairs
            .get(..self.len)?
            .iter()
            .find(|(candidate, _)| *candidate == name)
            .map(|&(_, value)| value)
    }
}

/// One operation matched from a request head.
#[derive(Clone, Copy)]
pub struct MatchedOperation<'a, T: 'static> {
    /// Matched registry entry, in the registry's own wrapper type.
    pub spec: &'static T,
    /// Borrowed path parameters.
    pub params: RouteParams<'a>,
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
pub fn match_operation<'a, T>(
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

    /// Two protocols sharing a matcher, declared without touching this module.
    const SPECS: &[OperationSpec] = &[
        OperationSpec {
            application_protocol: ApplicationProtocol::OPENAI_RESPONSES,
            operation_id: "createResponse",
            method: HttpMethod::Post,
            transport: Transport::Http,
            runtime_path: "/v1/responses",
            mode: HandlingMode::Transform,
            request_body: RequestBody::Json { required: true },
        },
        OperationSpec {
            application_protocol: ApplicationProtocol::OPENAI_RESPONSES,
            operation_id: "getResponse",
            method: HttpMethod::Get,
            transport: Transport::Http,
            runtime_path: "/v1/responses/{response_id}",
            mode: HandlingMode::Transform,
            request_body: RequestBody::None,
        },
        OperationSpec {
            application_protocol: ApplicationProtocol::OPENAI_RESPONSES,
            operation_id: "getInputTokenCounts",
            method: HttpMethod::Get,
            transport: Transport::Http,
            runtime_path: "/v1/responses/input_tokens",
            mode: HandlingMode::Local,
            request_body: RequestBody::None,
        },
        OperationSpec {
            application_protocol: ApplicationProtocol::ANTHROPIC_MESSAGES,
            operation_id: "createMessage",
            method: HttpMethod::Post,
            transport: Transport::Http,
            runtime_path: "/v1/messages",
            mode: HandlingMode::Passthrough,
            request_body: RequestBody::Json { required: true },
        },
    ];

    #[test]
    fn one_matcher_serves_several_protocols() {
        let responses = match_operation(SPECS, "POST", "/v1/responses", Transport::Http).unwrap();
        assert_eq!(
            responses.spec.application_protocol,
            ApplicationProtocol::OPENAI_RESPONSES
        );

        let messages = match_operation(SPECS, "POST", "/v1/messages", Transport::Http).unwrap();
        assert_eq!(
            messages.spec.application_protocol,
            ApplicationProtocol::ANTHROPIC_MESSAGES,
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
        let matched = match_operation(SPECS, "GET", "/v1/responses/resp_123", Transport::Http).unwrap();
        assert_eq!(matched.spec.operation_id, "getResponse");
        assert_eq!(matched.params.get("response_id"), Some("resp_123"));
        assert_eq!(matched.params.get("missing"), None);
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
    fn protocol_identifiers_are_open_ended() {
        const CUSTOM: ApplicationProtocol = ApplicationProtocol::new("vendor_custom_api");
        assert_eq!(CUSTOM.as_str(), "vendor_custom_api");
        assert_eq!(ApplicationProtocol::OPENAI_RESPONSES.as_str(), "openai_responses");
        assert_eq!(ApplicationProtocol::ANTHROPIC_MESSAGES.as_str(), "anthropic_messages");
    }
}
