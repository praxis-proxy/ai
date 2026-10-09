// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Files operation registry.
//!
//! Operation identity comes from the request head — method, path, and protocol
//! headers — rather than from body heuristics, so a request is recognized
//! before any payload is read. That matters more here than elsewhere: the
//! upload is `multipart/form-data` and the download returns opaque bytes, so
//! neither can be classified by inspecting a JSON body.
//!
//! Praxis proxies the Files contract rather than owning it, so these operations
//! declare their runtime request-body shape without an
//! `OwnedOperationContract`. Operation IDs are the official ones from the
//! pinned OpenAI specification, reproduced verbatim including upstream's
//! casing.

use std::ops::Deref;

use crate::{
    openai::operation::OpenAiOperationSpec,
    operation::{
        ApplicationProtocol, HandlingMode, HttpMethod, OperationEntry, OperationSpec, RequestBody, RouteParams,
        Transport, match_operation,
    },
};

/// Application protocol these operations belong to.
///
/// Declared beside the registry that owns it, so registering a protocol never
/// edits a shared list.
const APPLICATION_PROTOCOL: ApplicationProtocol = ApplicationProtocol::new("openai_files");

/// Static metadata for one Files operation.
#[derive(Clone, Copy)]
pub struct FilesOperationSpec {
    /// Runtime operation.
    pub operation: FilesOperation,
    /// Shared operation metadata.
    pub definition: OpenAiOperationSpec,
}

impl Deref for FilesOperationSpec {
    type Target = OpenAiOperationSpec;

    fn deref(&self) -> &Self::Target {
        &self.definition
    }
}

impl OperationEntry for FilesOperationSpec {
    fn spec(&self) -> &OperationSpec {
        &self.definition.runtime
    }
}

/// Convert a registry body declaration into a runtime request-body shape.
macro_rules! request_body_shape {
    ([none]) => {
        RequestBody::None
    };
    ([required multipart]) => {
        RequestBody::Multipart { required: true }
    };
}

/// Declare each Files operation once and derive its runtime metadata.
macro_rules! files_operations {
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
        /// One Files operation recognized from the request head.
        #[derive(Clone, Copy, Debug, Eq, PartialEq)]
        pub enum FilesOperation {
            $(
                #[doc = concat!(stringify!($method), " /v1", $path)]
                $operation,
            )+
        }

        /// All Files operations recognized by Praxis.
        pub const OPERATION_SPECS: &[FilesOperationSpec] = &[
            $(
                FilesOperationSpec {
                    operation: FilesOperation::$operation,
                    definition: OpenAiOperationSpec {
                        runtime: OperationSpec {
                            application_protocol: APPLICATION_PROTOCOL,
                            operation_id: $operation_id,
                            method: HttpMethod::$method,
                            transport: Transport::$transport,
                            runtime_path: concat!("/v1", $path),
                            mode: HandlingMode::$mode,
                            request_body: request_body_shape!($body),
                        },
                        spec_path: $path,
                        #[cfg(any(feature = "openai-conversations", feature = "openai-responses-openapi"))]
                        owned_contract: None,
                    },
                },
            )+
        ];
    };
}

files_operations! {
    ListFiles {
        operation_id: "listFiles",
        method: Get,
        transport: Http,
        path: "/files",
        mode: Passthrough,
        body: [none],
    },
    CreateFile {
        operation_id: "createFile",
        method: Post,
        transport: Http,
        path: "/files",
        mode: Passthrough,
        body: [required multipart],
    },
    RetrieveFile {
        operation_id: "retrieveFile",
        method: Get,
        transport: Http,
        path: "/files/{file_id}",
        mode: Passthrough,
        body: [none],
    },
    DeleteFile {
        operation_id: "deleteFile",
        method: Delete,
        transport: Http,
        path: "/files/{file_id}",
        mode: Passthrough,
        body: [none],
    },
    DownloadFile {
        operation_id: "downloadFile",
        method: Get,
        transport: Http,
        path: "/files/{file_id}/content",
        mode: Passthrough,
        body: [none],
    },
}

/// One matched Files route.
#[derive(Clone, Copy)]
pub(crate) struct MatchedFilesRoute<'a> {
    /// Matched operation metadata.
    pub spec: &'static FilesOperationSpec,
    /// Borrowed path parameters, captured by the shared matcher.
    pub(crate) params: RouteParams<'a>,
}

/// Return all Files operation specs.
#[must_use]
pub const fn operation_specs() -> &'static [FilesOperationSpec] {
    OPERATION_SPECS
}

/// Match a request head to a Files operation.
///
/// Files is reached over plain HTTP only.
pub(crate) fn match_route<'a>(method: &str, path: &'a str) -> Option<MatchedFilesRoute<'a>> {
    match_operation(OPERATION_SPECS, method, path, Transport::Http).map(|matched| MatchedFilesRoute {
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

    #[test]
    fn registry_keys_and_operation_ids_are_unique() {
        let keys = OPERATION_SPECS
            .iter()
            .map(|spec| (spec.method(), spec.transport().as_str(), spec.spec_path))
            .collect::<BTreeSet<_>>();
        assert_eq!(keys.len(), OPERATION_SPECS.len(), "duplicate method/transport/path key");

        let ids = OPERATION_SPECS
            .iter()
            .map(|spec| spec.operation_id())
            .collect::<BTreeSet<_>>();
        assert_eq!(ids.len(), OPERATION_SPECS.len(), "duplicate operation ID");
    }

    #[test]
    fn every_registered_operation_resolves_from_its_own_template() {
        for spec in OPERATION_SPECS {
            let path = spec.runtime_path().replace("{file_id}", "file_test");
            let matched = match_route(spec.method().as_str(), &path).unwrap();
            assert_eq!(
                matched.spec.operation,
                spec.operation,
                "{} {path} resolved to the wrong operation",
                spec.method().as_str()
            );
        }
    }

    /// The upload is `multipart/form-data`, so it has to be recognized from
    /// the head — there is no JSON body to classify.
    #[test]
    fn the_upload_declares_a_multipart_body() {
        let matched = match_route("POST", "/v1/files").unwrap();
        assert_eq!(matched.spec.operation, FilesOperation::CreateFile);
        assert!(
            matches!(matched.spec.request_body(), RequestBody::Multipart { required: true }),
            "createFile takes a required multipart body"
        );
    }

    /// A prefix match would let these through; explicit templates do not.
    #[test]
    fn unsupported_methods_and_unknown_subresources_do_not_match() {
        for (method, path) in [
            ("PUT", "/v1/files"),
            ("PATCH", "/v1/files/file_abc"),
            ("POST", "/v1/files/file_abc"),
            ("GET", "/v1/files/file_abc/unknown"),
            ("GET", "/v1/files/file_abc/content/extra"),
            ("GET", "/v1/files_extra"),
        ] {
            assert!(
                match_route(method, path).is_none(),
                "{method} {path} must not match a Files operation"
            );
        }
    }
}
