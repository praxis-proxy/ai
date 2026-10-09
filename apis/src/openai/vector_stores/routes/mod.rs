// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Vector Stores operation registry.
//!
//! Operation identity comes from the request head — method, path, and protocol
//! headers — rather than from body heuristics, so a request is recognized
//! before any payload is read.
//!
//! This family nests three levels deep and reuses `{file_id}` under a vector
//! store, where it names a vector-store file rather than a Files API upload.
//! Declaring every template explicitly is what keeps an unknown subresource
//! out of a matched route, which a path prefix cannot do.
//!
//! Praxis proxies the Vector Stores contract rather than owning it, so these
//! operations declare their runtime request-body shape without an
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
const APPLICATION_PROTOCOL: ApplicationProtocol = ApplicationProtocol::new("openai_vector_stores");

/// Static metadata for one Vector Stores operation.
#[derive(Clone, Copy)]
pub struct VectorStoresOperationSpec {
    /// Runtime operation.
    pub operation: VectorStoresOperation,
    /// Shared operation metadata.
    pub definition: OpenAiOperationSpec,
}

impl Deref for VectorStoresOperationSpec {
    type Target = OpenAiOperationSpec;

    fn deref(&self) -> &Self::Target {
        &self.definition
    }
}

impl OperationEntry for VectorStoresOperationSpec {
    fn spec(&self) -> &OperationSpec {
        &self.definition.runtime
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

/// Declare each Vector Stores operation once and derive its runtime metadata.
macro_rules! vector_stores_operations {
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
        /// One Vector Stores operation recognized from the request head.
        #[derive(Clone, Copy, Debug, Eq, PartialEq)]
        pub enum VectorStoresOperation {
            $(
                #[doc = concat!(stringify!($method), " /v1", $path)]
                $operation,
            )+
        }

        /// All Vector Stores operations recognized by Praxis.
        pub const OPERATION_SPECS: &[VectorStoresOperationSpec] = &[
            $(
                VectorStoresOperationSpec {
                    operation: VectorStoresOperation::$operation,
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

vector_stores_operations! {
    ListVectorStores {
        operation_id: "listVectorStores",
        method: Get,
        transport: Http,
        path: "/vector_stores",
        mode: Passthrough,
        body: [none],
    },
    CreateVectorStore {
        operation_id: "createVectorStore",
        method: Post,
        transport: Http,
        path: "/vector_stores",
        mode: Passthrough,
        body: [required json],
    },
    GetVectorStore {
        operation_id: "getVectorStore",
        method: Get,
        transport: Http,
        path: "/vector_stores/{vector_store_id}",
        mode: Passthrough,
        body: [none],
    },
    ModifyVectorStore {
        operation_id: "modifyVectorStore",
        method: Post,
        transport: Http,
        path: "/vector_stores/{vector_store_id}",
        mode: Passthrough,
        body: [required json],
    },
    DeleteVectorStore {
        operation_id: "deleteVectorStore",
        method: Delete,
        transport: Http,
        path: "/vector_stores/{vector_store_id}",
        mode: Passthrough,
        body: [none],
    },
    SearchVectorStore {
        operation_id: "searchVectorStore",
        method: Post,
        transport: Http,
        path: "/vector_stores/{vector_store_id}/search",
        mode: Passthrough,
        body: [required json],
    },
    CreateVectorStoreFileBatch {
        operation_id: "createVectorStoreFileBatch",
        method: Post,
        transport: Http,
        path: "/vector_stores/{vector_store_id}/file_batches",
        mode: Passthrough,
        body: [required json],
    },
    GetVectorStoreFileBatch {
        operation_id: "getVectorStoreFileBatch",
        method: Get,
        transport: Http,
        path: "/vector_stores/{vector_store_id}/file_batches/{batch_id}",
        mode: Passthrough,
        body: [none],
    },
    CancelVectorStoreFileBatch {
        operation_id: "cancelVectorStoreFileBatch",
        method: Post,
        transport: Http,
        path: "/vector_stores/{vector_store_id}/file_batches/{batch_id}/cancel",
        mode: Passthrough,
        body: [none],
    },
    ListFilesInVectorStoreBatch {
        operation_id: "listFilesInVectorStoreBatch",
        method: Get,
        transport: Http,
        path: "/vector_stores/{vector_store_id}/file_batches/{batch_id}/files",
        mode: Passthrough,
        body: [none],
    },
    ListVectorStoreFiles {
        operation_id: "listVectorStoreFiles",
        method: Get,
        transport: Http,
        path: "/vector_stores/{vector_store_id}/files",
        mode: Passthrough,
        body: [none],
    },
    CreateVectorStoreFile {
        operation_id: "createVectorStoreFile",
        method: Post,
        transport: Http,
        path: "/vector_stores/{vector_store_id}/files",
        mode: Passthrough,
        body: [required json],
    },
    GetVectorStoreFile {
        operation_id: "getVectorStoreFile",
        method: Get,
        transport: Http,
        path: "/vector_stores/{vector_store_id}/files/{file_id}",
        mode: Passthrough,
        body: [none],
    },
    UpdateVectorStoreFileAttributes {
        operation_id: "updateVectorStoreFileAttributes",
        method: Post,
        transport: Http,
        path: "/vector_stores/{vector_store_id}/files/{file_id}",
        mode: Passthrough,
        body: [required json],
    },
    DeleteVectorStoreFile {
        operation_id: "deleteVectorStoreFile",
        method: Delete,
        transport: Http,
        path: "/vector_stores/{vector_store_id}/files/{file_id}",
        mode: Passthrough,
        body: [none],
    },
    RetrieveVectorStoreFileContent {
        operation_id: "retrieveVectorStoreFileContent",
        method: Get,
        transport: Http,
        path: "/vector_stores/{vector_store_id}/files/{file_id}/content",
        mode: Passthrough,
        body: [none],
    },
}

/// One matched Vector Stores route.
#[derive(Clone, Copy)]
pub(crate) struct MatchedVectorStoresRoute<'a> {
    /// Matched operation metadata.
    pub spec: &'static VectorStoresOperationSpec,
    /// Borrowed path parameters, captured by the shared matcher.
    pub(crate) params: RouteParams<'a>,
}

/// Return all Vector Stores operation specs.
#[must_use]
pub const fn operation_specs() -> &'static [VectorStoresOperationSpec] {
    OPERATION_SPECS
}

/// Match a request head to a Vector Stores operation.
///
/// Vector Stores is reached over plain HTTP only.
pub(crate) fn match_route<'a>(method: &str, path: &'a str) -> Option<MatchedVectorStoresRoute<'a>> {
    match_operation(OPERATION_SPECS, method, path, Transport::Http).map(|matched| MatchedVectorStoresRoute {
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
            let path = spec
                .runtime_path()
                .replace("{vector_store_id}", "vs_test")
                .replace("{batch_id}", "vsfb_test")
                .replace("{file_id}", "file_test");
            let matched = match_route(spec.method().as_str(), &path).unwrap();
            assert_eq!(
                matched.spec.operation,
                spec.operation,
                "{} {path} resolved to the wrong operation",
                spec.method().as_str()
            );
        }
    }

    /// Three sibling templates share the `files/{file_id}` prefix and differ
    /// only by method or by one trailing segment, which is exactly what a
    /// prefix route cannot separate.
    #[test]
    fn sibling_templates_resolve_to_distinct_operations() {
        use VectorStoresOperation as Op;

        for (method, path, expected) in [
            ("GET", "/v1/vector_stores/vs_1/files/file_1", Op::GetVectorStoreFile),
            (
                "POST",
                "/v1/vector_stores/vs_1/files/file_1",
                Op::UpdateVectorStoreFileAttributes,
            ),
            (
                "DELETE",
                "/v1/vector_stores/vs_1/files/file_1",
                Op::DeleteVectorStoreFile,
            ),
            (
                "GET",
                "/v1/vector_stores/vs_1/files/file_1/content",
                Op::RetrieveVectorStoreFileContent,
            ),
            ("GET", "/v1/vector_stores/vs_1/files", Op::ListVectorStoreFiles),
        ] {
            let matched = match_route(method, path);
            assert!(matched.is_some(), "{method} {path} should match");
            assert_eq!(matched.unwrap().spec.operation, expected, "{method} {path}");
        }
    }

    /// A prefix match would admit all of these.
    #[test]
    fn unsupported_methods_and_unknown_subresources_do_not_match() {
        for (method, path) in [
            ("PUT", "/v1/vector_stores"),
            ("DELETE", "/v1/vector_stores"),
            ("GET", "/v1/vector_stores/vs_1/unknown"),
            ("GET", "/v1/vector_stores/vs_1/files/file_1/unknown"),
            ("POST", "/v1/vector_stores/vs_1/file_batches/batch_1/files"),
            ("GET", "/v1/vector_stores/vs_1/file_batches/batch_1/cancel"),
            ("GET", "/v1/vector_stores_extra"),
            ("GET", "/v1/vector_stores/vs_1/files/file_1/content/extra"),
        ] {
            assert!(
                match_route(method, path).is_none(),
                "{method} {path} must not match a Vector Stores operation"
            );
        }
    }
}
