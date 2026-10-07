// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Anthropic Messages operation registry.
//!
//! Operation identity comes from the request head — method, path, and transport
//! — rather than from body heuristics, so a request is recognized before any
//! payload is read. A Chat Completions-shaped, malformed, empty, or absent body
//! cannot change which operation the same head resolves to.
//!
//! # Source of truth
//!
//! The inventory below is transcribed from the pinned `OpenAPI` document vendored
//! at `docs/conformance/specs/anthropic-spec.json`, whose upstream origin and
//! digests are recorded in `docs/conformance/specs/anthropic-spec-source.json`.
//! Methods, paths, operation IDs, and request-body shapes are reproduced
//! verbatim from that document, including upstream's `snake_case` operation IDs.
//!
//! Scope is the non-beta `/v1/messages` surface, matching the scope
//! `docs/conformance/README.md` already declares for Anthropic review. The
//! other Anthropic surfaces (`/v1/complete`, `/v1/models`, `/v1/files`, and the
//! platform APIs) are deliberately absent: Praxis does not handle them, so they
//! publish no operation match.
//!
//! # Beta variants are a header concern, not a path
//!
//! The pinned document lists each beta variant under a path key suffixed
//! `?beta=true` — for example `GET /v1/messages/batches?beta=true` with
//! operation ID `beta_message_batches_list`. That suffix is a key the
//! mock-server document uses to keep both variants distinct within one
//! `OpenAPI` file; it is not a request parameter. Neither variant declares a
//! `beta` query parameter: they differ by the `anthropic-beta` request header.
//!
//! So no real request path carries `?beta=true`, and registering such entries
//! would both encode that artifact and still miss real beta traffic, which
//! arrives on the non-beta path with a header. The beta operation IDs are
//! therefore not registered, and beta selection is not part of operation
//! identity — the same treatment `anthropic-version` already gets.
//!
//! One consequence is worth stating plainly: because query strings do not
//! affect matching, a request to `/v1/messages/batches?beta=true` resolves to
//! `message_batches_list`, the non-beta operation its method and path name.
//! Distinguishing beta behavior is left to Anthropic request processing and the
//! backend, which can read the header.
//!
//! That document is a pinned reference, not a wired conformance gate — there is
//! no `oasdiff` or capability projection for Anthropic, so this registry makes
//! no machine-checked specification-conformance claim.
//!
//! # Handling modes
//!
//! [`HandlingMode`] describes what Praxis does at its boundary, not which chain
//! a deployment assembles. `createMessage` is [`HandlingMode::Inspect`] because
//! every supported chain reads selected fields from it while preserving the
//! forwarded payload; whether a chain additionally rewrites it for a Chat
//! Completions backend is pipeline configuration rather than operation
//! identity. The batch and token-counting operations are
//! [`HandlingMode::Passthrough`]: no Anthropic filter deliberately targets
//! them.

use crate::operation::{
    ApplicationProtocol, HandlingMode, HttpMethod, OperationEntry, OperationSpec, RequestBody, RouteParams, Transport,
    match_operation,
};

/// Application protocol these operations belong to.
///
/// Declared beside the registry that owns it, so registering a protocol
/// never edits a shared list.
const APPLICATION_PROTOCOL: ApplicationProtocol = ApplicationProtocol::new("anthropic_messages");

/// Static metadata for one Anthropic Messages operation.
///
/// Holds the crate's shared `OperationSpec` directly rather than wrapping it in a
/// provider type: Praxis generates no `OpenAPI` contract for Anthropic, so
/// there is no provider-owned metadata to carry beside the runtime identity.
#[derive(Clone, Copy)]
pub struct AnthropicMessagesOperationSpec {
    /// Runtime operation.
    pub operation: AnthropicMessagesOperation,
    /// Shared operation metadata.
    ///
    /// Crate-private because the shared `OperationSpec` is: external callers read this
    /// entry through the accessors below rather than the shared struct.
    pub(crate) definition: OperationSpec,
}

impl OperationEntry for AnthropicMessagesOperationSpec {
    fn spec(&self) -> &OperationSpec {
        &self.definition
    }
}

impl AnthropicMessagesOperationSpec {
    /// Stable operation ID, as spelled in the pinned document.
    #[must_use]
    pub const fn operation_id(&self) -> &'static str {
        self.definition.operation_id
    }

    /// Typed HTTP method.
    #[must_use]
    pub const fn method(&self) -> HttpMethod {
        self.definition.method
    }

    /// Application protocol that owns the operation.
    #[must_use]
    pub const fn application_protocol(&self) -> ApplicationProtocol {
        self.definition.application_protocol
    }

    /// Transport the operation is reached over.
    #[must_use]
    pub const fn transport(&self) -> Transport {
        self.definition.transport
    }

    /// Runtime path template handled by Praxis.
    ///
    /// Equal to the path in the pinned document: Anthropic's paths already
    /// carry the `/v1` prefix Praxis serves them on.
    #[must_use]
    pub const fn runtime_path(&self) -> &'static str {
        self.definition.runtime_path
    }

    /// Runtime request-body shape.
    #[must_use]
    pub const fn request_body(&self) -> RequestBody {
        self.definition.request_body
    }

    /// Proxy handling mode.
    #[must_use]
    pub const fn mode(&self) -> HandlingMode {
        self.definition.mode
    }
}

/// Convert a registry body declaration into a runtime request-body shape.
macro_rules! request_body_shape {
    ([none]) => {
        RequestBody::None
    };
    ([required json]) => {
        RequestBody::Json { required: true }
    };
}

/// Declare each Anthropic Messages operation once and derive its metadata.
macro_rules! anthropic_messages_operations {
    (
        $(
            $operation:ident {
                operation_id: $operation_id:literal,
                method: $method:ident,
                transport: $transport:ident,
                path: $path:literal,
                mode: $mode:ident,
                body: $body:tt $(,)?
            }
        ),+ $(,)?
    ) => {
        /// One Anthropic Messages operation recognized from the request head.
        #[derive(Clone, Copy, Debug, Eq, PartialEq)]
        pub enum AnthropicMessagesOperation {
            $(
                #[doc = concat!(stringify!($method), " ", $path)]
                $operation,
            )+
        }

        /// All Anthropic Messages operations recognized by Praxis.
        pub const OPERATION_SPECS: &[AnthropicMessagesOperationSpec] = &[
            $(
                AnthropicMessagesOperationSpec {
                    operation: AnthropicMessagesOperation::$operation,
                    definition: OperationSpec {
                        application_protocol: APPLICATION_PROTOCOL,
                        operation_id: $operation_id,
                        method: HttpMethod::$method,
                        transport: Transport::$transport,
                        runtime_path: $path,
                        mode: HandlingMode::$mode,
                        request_body: request_body_shape!($body),
                    },
                },
            )+
        ];
    };
}

anthropic_messages_operations! {
    CreateMessage {
        operation_id: "messages_post",
        method: Post,
        transport: Http,
        path: "/v1/messages",
        mode: Inspect,
        body: [required json],
    },
    CountMessageTokens {
        operation_id: "messages_count_tokens_post",
        method: Post,
        transport: Http,
        path: "/v1/messages/count_tokens",
        mode: Passthrough,
        body: [required json],
    },
    CreateMessageBatch {
        operation_id: "message_batches_post",
        method: Post,
        transport: Http,
        path: "/v1/messages/batches",
        mode: Passthrough,
        body: [required json],
    },
    ListMessageBatches {
        operation_id: "message_batches_list",
        method: Get,
        transport: Http,
        path: "/v1/messages/batches",
        mode: Passthrough,
        body: [none],
    },
    GetMessageBatch {
        operation_id: "message_batches_retrieve",
        method: Get,
        transport: Http,
        path: "/v1/messages/batches/{message_batch_id}",
        mode: Passthrough,
        body: [none],
    },
    DeleteMessageBatch {
        operation_id: "message_batches_delete",
        method: Delete,
        transport: Http,
        path: "/v1/messages/batches/{message_batch_id}",
        mode: Passthrough,
        body: [none],
    },
    CancelMessageBatch {
        operation_id: "message_batches_cancel",
        method: Post,
        transport: Http,
        path: "/v1/messages/batches/{message_batch_id}/cancel",
        mode: Passthrough,
        body: [none],
    },
    GetMessageBatchResults {
        operation_id: "message_batches_results",
        method: Get,
        transport: Http,
        path: "/v1/messages/batches/{message_batch_id}/results",
        mode: Passthrough,
        body: [none],
    },
}

/// One matched Anthropic Messages route.
#[derive(Clone, Copy)]
pub(crate) struct MatchedAnthropicMessagesRoute<'a> {
    /// Matched operation metadata.
    pub spec: &'static AnthropicMessagesOperationSpec,
    /// Borrowed path parameters, captured by the shared matcher.
    pub(crate) params: RouteParams<'a>,
}

/// Return all Anthropic Messages operation specs.
#[must_use]
pub const fn operation_specs() -> &'static [AnthropicMessagesOperationSpec] {
    OPERATION_SPECS
}

/// Match a request head to an Anthropic Messages operation.
///
/// Anthropic Messages is reached over plain HTTP only. Required headers such as
/// `anthropic-version` are contract validation left to request processing and
/// the backend: method and path already identify the operation, so a missing
/// version header must not change which operation is selected.
pub(crate) fn match_route<'a>(method: &str, path: &'a str) -> Option<MatchedAnthropicMessagesRoute<'a>> {
    match_operation(OPERATION_SPECS, method, path, Transport::Http).map(|matched| MatchedAnthropicMessagesRoute {
        spec: matched.spec,
        params: matched.params,
    })
}

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::unwrap_used, reason = "tests")]
mod tests {
    use std::collections::BTreeSet;

    use super::*;

    /// Stand-in batch identifier used when filling path templates.
    const BATCH_ID: &str = "msgbatch_abc123";

    #[test]
    fn registry_keys_and_operation_ids_are_unique() {
        let keys = OPERATION_SPECS
            .iter()
            .map(|spec| (spec.method(), spec.transport().as_str(), spec.runtime_path()))
            .collect::<BTreeSet<_>>();
        assert_eq!(keys.len(), OPERATION_SPECS.len(), "duplicate method/transport/path key");

        let ids = OPERATION_SPECS
            .iter()
            .map(AnthropicMessagesOperationSpec::operation_id)
            .collect::<BTreeSet<_>>();
        assert_eq!(ids.len(), OPERATION_SPECS.len(), "duplicate operation ID");
    }

    #[test]
    fn every_registered_operation_resolves_from_its_own_template() {
        for spec in OPERATION_SPECS {
            let path = spec.runtime_path().replace("{message_batch_id}", BATCH_ID);
            let matched = match_route(spec.method().as_str(), &path).unwrap();
            assert_eq!(
                matched.spec.operation,
                spec.operation,
                "{} {path} resolved to the wrong operation",
                spec.method().as_str()
            );
        }
    }

    #[test]
    fn every_operation_belongs_to_the_anthropic_messages_protocol() {
        assert!(
            OPERATION_SPECS
                .iter()
                .all(|spec| spec.application_protocol().as_str() == "anthropic_messages"),
            "the registry declares one protocol for every operation it owns"
        );
    }

    #[test]
    fn a_literal_segment_outranks_a_batch_identifier() {
        let tokens = match_route("POST", "/v1/messages/count_tokens").unwrap();
        assert_eq!(
            tokens.spec.operation,
            AnthropicMessagesOperation::CountMessageTokens,
            "count_tokens must not be consumed as a subresource of /v1/messages"
        );

        let batches = match_route("GET", "/v1/messages/batches").unwrap();
        assert_eq!(
            batches.spec.operation,
            AnthropicMessagesOperation::ListMessageBatches,
            "the batches collection must not be read as a batch identifier"
        );
    }

    #[test]
    fn identifier_paths_capture_the_batch_id() {
        let path = "/v1/messages/batches/msgbatch_abc123/results";
        let matched = match_route("GET", path).unwrap();
        assert_eq!(
            matched.spec.operation,
            AnthropicMessagesOperation::GetMessageBatchResults
        );
        assert_eq!(matched.params.get("message_batch_id"), Some(BATCH_ID));

        let offsets = matched.params.offsets_in(path).unwrap();
        assert_eq!(offsets.get(path, "message_batch_id"), Some(BATCH_ID));
    }

    #[test]
    fn method_separates_operations_sharing_one_path() {
        let created = match_route("POST", "/v1/messages/batches").unwrap();
        assert_eq!(created.spec.operation, AnthropicMessagesOperation::CreateMessageBatch);

        let listed = match_route("GET", "/v1/messages/batches").unwrap();
        assert_eq!(listed.spec.operation, AnthropicMessagesOperation::ListMessageBatches);

        let retrieved = match_route("GET", "/v1/messages/batches/msgbatch_abc123").unwrap();
        assert_eq!(retrieved.spec.operation, AnthropicMessagesOperation::GetMessageBatch);

        let deleted = match_route("DELETE", "/v1/messages/batches/msgbatch_abc123").unwrap();
        assert_eq!(deleted.spec.operation, AnthropicMessagesOperation::DeleteMessageBatch);
    }

    #[test]
    fn trailing_slash_and_query_string_are_normalized() {
        for path in [
            "/v1/messages",
            "/v1/messages/",
            "/v1/messages?stream=true",
            "/v1/messages/?stream=true",
        ] {
            let matched = match_route("POST", path).unwrap();
            assert_eq!(
                matched.spec.operation,
                AnthropicMessagesOperation::CreateMessage,
                "{path}"
            );
        }
    }

    /// A `?beta=true` query resolves to the non-beta operation.
    ///
    /// The pinned document uses that suffix as a path key to separate beta
    /// variants, not as a request parameter — the variants differ by the
    /// `anthropic-beta` header. Query strings do not affect matching, so such a
    /// request resolves to the operation its method and path name. Asserted
    /// rather than left implicit because the pinned document's path keys invite
    /// the opposite assumption.
    #[test]
    fn a_beta_query_resolves_to_the_non_beta_operation() {
        let listed = match_route("GET", "/v1/messages/batches?beta=true").unwrap();
        assert_eq!(listed.spec.operation, AnthropicMessagesOperation::ListMessageBatches);
        assert_eq!(listed.spec.operation_id(), "message_batches_list");

        let created = match_route("POST", "/v1/messages?beta=true").unwrap();
        assert_eq!(created.spec.operation_id(), "messages_post");
    }

    /// No beta operation ID from the pinned document is registered.
    #[test]
    fn beta_operation_ids_are_not_registered() {
        assert!(
            OPERATION_SPECS
                .iter()
                .all(|spec| !spec.operation_id().starts_with("beta_")),
            "beta variants are selected by header, so they are not separate registry entries"
        );
    }

    #[test]
    fn unsupported_methods_do_not_match() {
        for (method, path) in [
            ("GET", "/v1/messages"),
            ("DELETE", "/v1/messages"),
            ("PUT", "/v1/messages"),
            ("PATCH", "/v1/messages"),
            ("POST", "/v1/messages/batches/msgbatch_abc123"),
            ("DELETE", "/v1/messages/batches"),
            ("GET", "/v1/messages/count_tokens"),
        ] {
            assert!(
                match_route(method, path).is_none(),
                "{method} {path} must publish no operation match"
            );
        }
    }

    #[test]
    fn malformed_subresource_paths_do_not_match() {
        for path in [
            "/v1/messages/msg_abc123",
            "/v1/messages/batches/msgbatch_abc123/other",
            "/v1/messages/batches/msgbatch_abc123/results/extra",
            "/v1/messages//batches",
            "/messages",
        ] {
            assert!(
                match_route("GET", path).is_none(),
                "GET {path} must publish no operation match"
            );
        }
    }

    #[test]
    fn unregistered_anthropic_surfaces_do_not_match() {
        for (method, path) in [
            ("POST", "/v1/complete"),
            ("GET", "/v1/models"),
            ("GET", "/v1/models/claude-sonnet-4-5"),
            ("POST", "/v1/files"),
            ("GET", "/v1/skills"),
            ("GET", "/v1/organizations/me"),
        ] {
            assert!(
                match_route(method, path).is_none(),
                "{method} {path} is outside the registered Messages surface"
            );
        }
    }

    #[test]
    fn body_shapes_come_from_the_pinned_document() {
        for spec in OPERATION_SPECS {
            let expected = match spec.operation {
                AnthropicMessagesOperation::CreateMessage
                | AnthropicMessagesOperation::CountMessageTokens
                | AnthropicMessagesOperation::CreateMessageBatch => RequestBody::Json { required: true },
                _ => RequestBody::None,
            };
            assert_eq!(
                spec.request_body(),
                expected,
                "{:?} reported the wrong body shape",
                spec.operation
            );
        }
    }

    #[test]
    fn a_post_without_a_body_is_distinguishable_from_one_with_a_body() {
        let cancel = match_route("POST", "/v1/messages/batches/msgbatch_abc123/cancel").unwrap();
        assert_eq!(cancel.spec.operation, AnthropicMessagesOperation::CancelMessageBatch);
        assert_eq!(
            cancel.spec.request_body(),
            RequestBody::None,
            "cancel is a POST that takes no body, so body shape is not implied by method"
        );
        assert!(!cancel.spec.request_body().is_present());
    }

    #[test]
    fn create_message_is_the_only_inspected_operation() {
        for spec in OPERATION_SPECS {
            let expected = match spec.operation {
                AnthropicMessagesOperation::CreateMessage => HandlingMode::Inspect,
                _ => HandlingMode::Passthrough,
            };
            assert_eq!(
                spec.mode(),
                expected,
                "{:?} reported the wrong handling mode",
                spec.operation
            );
        }
    }

    #[test]
    fn operation_specs_exposes_the_registry() {
        assert_eq!(operation_specs().len(), OPERATION_SPECS.len());
    }

    #[test]
    fn transport_is_part_of_identity() {
        assert!(
            match_operation(OPERATION_SPECS, "POST", "/v1/messages", Transport::WebSocket).is_none(),
            "no Anthropic Messages operation is reached over a WebSocket upgrade"
        );
    }
}
