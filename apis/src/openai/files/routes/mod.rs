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
//! Each operation declares its owned contract (parameters, request body, and
//! response) once, and the runtime request-body shape is derived from that same
//! declaration so the two cannot diverge. Entries stay `Passthrough` until local
//! handlers land; generation and conformance comparison follow the declared
//! contract, not the handling mode. Operation IDs are the official ones
//! from the pinned OpenAI specification, reproduced verbatim including
//! upstream's casing.

use std::ops::Deref;

#[cfg(feature = "openai-files")]
use super::contracts::{
    CreateFileRequest, DeleteFileResponse, FileContent, ListFilesResponse, OpenAiFile, list_files_limit_schema,
    list_files_order_schema,
};
#[cfg(feature = "openai-files")]
use crate::openai::operation::{
    MediaTypeSpec, OwnedOperationContract, RequestBodySpec, ResponseSpec, path_parameter, query_parameter,
    schema_binding,
};
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

/// JSON media type used by Files metadata responses.
#[cfg(feature = "openai-files")]
const JSON_CONTENT_TYPE: &str = "application/json";

/// Multipart media type of the upload body.
#[cfg(feature = "openai-files")]
const MULTIPART_CONTENT_TYPE: &str = "multipart/form-data";

/// Binary media type of the download response.
#[cfg(feature = "openai-files")]
const BINARY_CONTENT_TYPE: &str = "application/octet-stream";

/// Convert a registry request declaration into an optional schema binding.
#[cfg(feature = "openai-files")]
macro_rules! request_binding {
    ([none]) => {
        None
    };
    ([required multipart $schema:ty]) => {
        Some(RequestBodySpec {
            required: true,
            content: &[MediaTypeSpec::new(MULTIPART_CONTENT_TYPE, schema_binding!($schema))],
        })
    };
}

/// Convert a registry response declaration into a media type binding.
#[cfg(feature = "openai-files")]
macro_rules! response_binding {
    ([json $schema:ty]) => {
        MediaTypeSpec::new(JSON_CONTENT_TYPE, schema_binding!($schema))
    };
    ([binary $schema:ty]) => {
        MediaTypeSpec::new(BINARY_CONTENT_TYPE, schema_binding!($schema))
    };
}

/// Convert a registry contract declaration into owned metadata.
#[cfg(any(
    feature = "openai-files",
    feature = "openai-conversations",
    feature = "openai-responses-openapi"
))]
macro_rules! operation_contract {
    (
        {
            parameters: [$($parameter:expr),* $(,)?],
            request: $request:tt,
            response: $response:tt $(,)?
        }
    ) => {{
        #[cfg(feature = "openai-files")]
        {
            Some(OwnedOperationContract {
                parameters: &[$($parameter),*],
                request: request_binding!($request),
                responses: &[ResponseSpec {
                    status: "200",
                    description: "OK",
                    content: &[response_binding!($response)],
                }],
            })
        }
        #[cfg(not(feature = "openai-files"))]
        {
            None
        }
    }};
}

/// Derive the runtime request-body shape from a registry request declaration.
macro_rules! request_body_shape {
    ([none]) => {
        RequestBody::None
    };
    ([required multipart $schema:ty]) => {
        RequestBody::Multipart { required: true }
    };
}

/// Derive the runtime request-body shape from a contract declaration.
///
/// Reads the same `request:` token as `operation_contract!`, so the body shape
/// and the generated contract cannot drift apart.
macro_rules! contract_request_body {
    ({ parameters: [$($parameter:expr),* $(,)?],request: $request:tt,response: $response:tt $(,)? }) => {
        request_body_shape!($request)
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
                contract: $contract:tt $(,)?
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
                            request_body: contract_request_body!($contract),
                        },
                        spec_path: $path,
                        #[cfg(any(
                            feature = "openai-conversations",
                            feature = "openai-files",
                            feature = "openai-responses-openapi"
                        ))]
                        owned_contract: operation_contract!($contract),
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
        contract: {
            parameters: [
                query_parameter!("purpose", String, "Only return files with the given purpose."),
                query_parameter!(
                    "limit",
                    schema_fn = list_files_limit_schema,
                    "A limit on the number of objects to be returned. Limit can range between 1 and 10,000, and the default is 10,000."
                ),
                query_parameter!(
                    "order",
                    schema_fn = list_files_order_schema,
                    "Sort order by the `created_at` timestamp of the objects. `asc` for ascending order and `desc` for descending order."
                ),
                query_parameter!(
                    "after",
                    String,
                    "A cursor for use in pagination. `after` is an object ID that defines your place in the list."
                ),
            ],
            request: [none],
            response: [json ListFilesResponse],
        },
    },
    CreateFile {
        operation_id: "createFile",
        method: Post,
        transport: Http,
        path: "/files",
        mode: Passthrough,
        contract: {
            parameters: [],
            request: [required multipart CreateFileRequest],
            response: [json OpenAiFile],
        },
    },
    RetrieveFile {
        operation_id: "retrieveFile",
        method: Get,
        transport: Http,
        path: "/files/{file_id}",
        mode: Passthrough,
        contract: {
            parameters: [path_parameter!("file_id", "The ID of the file to use for this request.")],
            request: [none],
            response: [json OpenAiFile],
        },
    },
    DeleteFile {
        operation_id: "deleteFile",
        method: Delete,
        transport: Http,
        path: "/files/{file_id}",
        mode: Passthrough,
        contract: {
            parameters: [path_parameter!("file_id", "The ID of the file to use for this request.")],
            request: [none],
            response: [json DeleteFileResponse],
        },
    },
    DownloadFile {
        operation_id: "downloadFile",
        method: Get,
        transport: Http,
        path: "/files/{file_id}/content",
        mode: Passthrough,
        contract: {
            parameters: [path_parameter!("file_id", "The ID of the file to use for this request.")],
            request: [none],
            response: [binary FileContent],
        },
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

    #[cfg(feature = "openai-files")]
    mod contract {
        use super::*;

        #[test]
        fn every_spec_declares_a_contract_but_passthrough_does_not_own_it() {
            for spec in OPERATION_SPECS {
                assert!(
                    spec.owned_contract().is_some(),
                    "{} lacks a contract",
                    spec.operation_id()
                );
                assert!(!spec.owns_contract(), "{} is passthrough", spec.operation_id());
            }
        }

        #[test]
        fn declared_request_body_shape_matches_generated_contract() {
            for spec in OPERATION_SPECS {
                let contract_has_body = spec.owned_contract().is_some_and(|contract| contract.request.is_some());
                assert_eq!(
                    spec.request_body().is_present(),
                    contract_has_body,
                    "request_body shape drifted from the contract for {:?}",
                    spec.operation
                );
                if let Some(request) = spec.owned_contract().and_then(|contract| contract.request) {
                    assert_eq!(spec.request_body().is_required(), request.required);
                }
            }
        }
    }
}
