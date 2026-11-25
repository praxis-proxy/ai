// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! [`OpenaiFilesFilter`] serves `GET /v1/files` and `GET /v1/files/{file_id}`
//! locally via `FilterAction::TerminalResponse`.
//!
//! The filter classifies from the request head with the Files route registry,
//! so it works without `ai_operation`. Upload, content and delete are not
//! implemented locally yet and pass through unchanged.

use std::sync::Arc;

use async_trait::async_trait;
use praxis_filter::{FilterAction, FilterError, HttpFilter, HttpFilterContext, TerminalResponse, parse_filter_config};
use tracing::{debug, warn};

use super::{
    FileList, FileService, FileServiceError,
    config::FilesConfig,
    contracts::ListFilesResponse,
    handlers,
    routes::{FilesOperation, MatchedFilesRoute, match_route},
};
use crate::state_owner::{StateOwner, require_state_owner};

// -----------------------------------------------------------------------------
// OpenaiFilesFilter
// -----------------------------------------------------------------------------

/// Handles the local Files metadata endpoints.
///
/// `GET /v1/files` and `GET /v1/files/{file_id}` are answered from the
/// owner-scoped file catalog and never reach upstream. Every other request,
/// including the not yet implemented upload, content and delete operations,
/// continues down the pipeline.
///
/// A trusted owner is required: place `state_owner` earlier in the chain. A
/// request without one is rejected with 401. Missing, expired, pending,
/// failed, deleted and other tenants' files all return the same 404. Query
/// errors return 400 and catalog failures a generic 500; catalog error text
/// never reaches the client or the logs.
///
/// # YAML
///
/// ```yaml
/// - filter: state_owner
///   mode: single_tenant
///   tenant_id: development
/// - filter: openai_files
/// ```
#[derive(Default)]
pub struct OpenaiFilesFilter {
    /// Service answering the local operations; `None` until a catalog is
    /// provisioned, which yields a 503.
    service: Option<Arc<dyn FileService>>,
}

impl OpenaiFilesFilter {
    /// Create a filter from parsed YAML config.
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] if the YAML config is invalid.
    pub fn from_config(config: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
        let _config: FilesConfig = parse_filter_config("openai_files", config)?;
        Ok(Box::new(Self::default()))
    }

    /// Create a filter that answers from `service`.
    #[cfg(test)]
    pub(crate) fn with_service(service: Arc<dyn FileService>) -> Self {
        Self { service: Some(service) }
    }

    /// Resolve the service for this request.
    fn resolve_service(&self, _ctx: &HttpFilterContext<'_>) -> Option<Arc<dyn FileService>> {
        self.service.clone()
    }
}

#[async_trait]
impl HttpFilter for OpenaiFilesFilter {
    fn name(&self) -> &'static str {
        "openai_files"
    }

    async fn on_request(&self, ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        let path = ctx.request.uri.path();
        let Some(route) = match_route(ctx.request.method.as_str(), path) else {
            return Ok(FilterAction::Continue);
        };
        let operation = route.spec.operation;
        // Temporary: upload, content and delete (#500, #501, delete PR) pass
        // through without an owner. Delete this guard when the last of them is
        // served locally; every matched route then needs the owner below.
        if matches!(
            operation,
            FilesOperation::CreateFile | FilesOperation::DeleteFile | FilesOperation::DownloadFile
        ) {
            return Ok(FilterAction::Continue);
        }
        let owner = match require_state_owner(ctx) {
            Ok(owner) => owner,
            Err(action) => return Ok(action),
        };
        let Some(service) = self.resolve_service(ctx) else {
            warn!(operation = ?operation, status = 503_u16, "files service unavailable");
            return terminal(handlers::unavailable_response());
        };
        serve(service.as_ref(), owner, &route, ctx.request.uri.query()).await
    }
}

// -----------------------------------------------------------------------------
// Dispatch
// -----------------------------------------------------------------------------

/// Serve a matched route that is implemented locally.
async fn serve(
    service: &dyn FileService,
    owner: &StateOwner,
    route: &MatchedFilesRoute<'_>,
    query: Option<&str>,
) -> Result<FilterAction, FilterError> {
    let response = match route.spec.operation {
        FilesOperation::RetrieveFile => {
            let id = route
                .params
                .get("file_id")
                .ok_or_else(|| FilterError::from("openai_files: matched retrieve route missing file id"))?;
            retrieve(service, owner, id).await
        },
        FilesOperation::ListFiles => list(service, owner, query).await,
        FilesOperation::CreateFile | FilesOperation::DeleteFile | FilesOperation::DownloadFile => {
            return Ok(FilterAction::Continue);
        },
    };
    terminal(response)
}

/// Wrap a rendered response as the filter's terminal action.
fn terminal(response: Result<TerminalResponse, FilterError>) -> Result<FilterAction, FilterError> {
    response.map(|response| FilterAction::TerminalResponse(Box::new(response)))
}

/// Serve `retrieveFile`.
async fn retrieve(service: &dyn FileService, owner: &StateOwner, id: &str) -> Result<TerminalResponse, FilterError> {
    match service.retrieve(owner, id).await {
        Ok(info) => {
            debug!(operation = "retrieveFile", status = 200_u16, "files request served");
            handlers::json_response(200, &super::contracts::OpenAiFile::from(info))
        },
        Err(error) => {
            log_failure("retrieveFile", &error);
            handlers::error_response(&error, Some(id))
        },
    }
}

/// Serve `listFiles`.
async fn list(
    service: &dyn FileService,
    owner: &StateOwner,
    query: Option<&str>,
) -> Result<TerminalResponse, FilterError> {
    let query = match handlers::parse_list_query(query) {
        Ok(query) => query,
        Err(invalid) => {
            debug!(operation = "listFiles", status = 400_u16, "files query rejected");
            return handlers::invalid_query_response(&invalid);
        },
    };
    match service.list(owner, &FileList::from(query)).await {
        Ok(page) => {
            debug!(
                operation = "listFiles",
                status = 200_u16,
                has_more = page.has_more,
                page_size = page.data.len(),
                "files request served"
            );
            handlers::json_response(200, &ListFilesResponse::from(page))
        },
        Err(error) => {
            log_failure("listFiles", &error);
            handlers::error_response(&error, None)
        },
    }
}

/// Log a failed operation with the service error's `Display` text, which is
/// a fixed phrase per variant.
fn log_failure(operation: &'static str, error: &FileServiceError) {
    if matches!(error, FileServiceError::NotFound | FileServiceError::InvalidInput(_)) {
        debug!(operation, error = %error, "files request failed");
    } else {
        warn!(operation, error = %error, "files request failed");
    }
}

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::unimplemented,
    reason = "tests"
)]
mod tests {
    use http::Method;
    use praxis_ai_store::CatalogError;
    use serde_json::{Value, json};

    use super::*;
    use crate::{
        openai::files::{FileDownload, FileInfo, FilePage, FilePurpose, FileUploadSession},
        test_utils::{capture_log_output, make_filter_context, make_request},
    };

    /// What the fake service returns for retrieve and list.
    #[derive(Clone, Copy)]
    enum Outcome {
        Found,
        Missing,
        InvalidCursor,
        CatalogDown,
    }

    struct FakeService(Outcome);

    fn info(filename: &str) -> FileInfo {
        FileInfo {
            id: "file_1".to_owned(),
            bytes: 3,
            created_at: 10,
            expires_at: None,
            filename: filename.to_owned(),
            purpose: FilePurpose::Assistants,
        }
    }

    impl FakeService {
        fn outcome<T>(&self, found: T) -> Result<T, FileServiceError> {
            match self.0 {
                Outcome::Found => Ok(found),
                Outcome::Missing => Err(FileServiceError::NotFound),
                Outcome::InvalidCursor => Err(CatalogError::InvalidCursor.into()),
                Outcome::CatalogDown => Err(CatalogError::Backend("secret-dsn".into()).into()),
            }
        }
    }

    #[async_trait]
    impl FileService for FakeService {
        async fn begin_upload(
            &self,
            _owner: &StateOwner,
            _filename: String,
            _size_hint: Option<u64>,
        ) -> Result<Box<dyn FileUploadSession + 'static>, FileServiceError> {
            unimplemented!()
        }

        async fn retrieve(&self, _owner: &StateOwner, _id: &str) -> Result<FileInfo, FileServiceError> {
            self.outcome(info("marker-filename"))
        }

        async fn list(&self, _owner: &StateOwner, _options: &FileList) -> Result<FilePage, FileServiceError> {
            self.outcome(FilePage {
                data: vec![info("marker-filename")],
                has_more: false,
            })
        }

        async fn content(&self, _owner: &StateOwner, _id: &str) -> Result<FileDownload, FileServiceError> {
            unimplemented!()
        }

        async fn delete(&self, _owner: &StateOwner, _id: &str) -> Result<(), FileServiceError> {
            unimplemented!()
        }
    }

    fn filter(outcome: Outcome) -> OpenaiFilesFilter {
        OpenaiFilesFilter::with_service(Arc::new(FakeService(outcome)))
    }

    /// Run the filter; `owned` installs a trusted owner first.
    async fn run(filter: &OpenaiFilesFilter, method: Method, uri: &str, owned: bool) -> FilterAction {
        let request = make_request(method, uri);
        let mut ctx = make_filter_context(&request);
        if owned {
            ctx.extensions
                .insert(StateOwner::from_trusted_parts("tenant", "issuer", "subject").unwrap());
        }
        filter.on_request(&mut ctx).await.unwrap()
    }

    fn terminal_parts(action: FilterAction) -> (u16, Value) {
        let FilterAction::TerminalResponse(response) = action else {
            panic!("expected a terminal response, got {action:?}");
        };
        (
            response.status,
            serde_json::from_slice(response.body.as_ref().unwrap()).unwrap(),
        )
    }

    #[tokio::test]
    async fn retrieve_and_list_are_served_locally() {
        let (status, body) = terminal_parts(run(&filter(Outcome::Found), Method::GET, "/v1/files/file_1", true).await);
        assert_eq!(status, 200);
        assert_eq!(body["id"], "file_1");
        assert_eq!(body["status"], "processed");

        let (status, body) = terminal_parts(run(&filter(Outcome::Found), Method::GET, "/v1/files?limit=5", true).await);
        assert_eq!(status, 200);
        assert_eq!(body["object"], "list");
        assert_eq!(body["first_id"], "file_1");
        assert_eq!(body["has_more"], false);
    }

    #[tokio::test]
    async fn missing_owner_is_401_and_never_reaches_the_service() {
        for uri in ["/v1/files", "/v1/files/file_1"] {
            let action = run(&filter(Outcome::Found), Method::GET, uri, false).await;
            let FilterAction::Reject(rejection) = action else {
                panic!("expected a rejection for {uri}");
            };
            assert_eq!(rejection.status, 401);
        }
    }

    #[tokio::test]
    async fn service_errors_map_to_statuses() {
        let cases = [
            (Outcome::Missing, "/v1/files/file_1", 404, "invalid_request_error"),
            (
                Outcome::InvalidCursor,
                "/v1/files?after=x",
                400,
                "invalid_request_error",
            ),
            (Outcome::CatalogDown, "/v1/files/file_1", 500, "server_error"),
            (Outcome::CatalogDown, "/v1/files", 500, "server_error"),
        ];
        for (outcome, uri, expected, error_type) in cases {
            let (status, body) = terminal_parts(run(&filter(outcome), Method::GET, uri, true).await);
            assert_eq!(status, expected, "{uri}");
            assert_eq!(body["error"]["type"], error_type);
            assert!(!body.to_string().contains("secret-dsn"));
        }
    }

    #[tokio::test]
    async fn invalid_limit_is_400_naming_the_field() {
        for query in ["limit=0", "limit=10001"] {
            let uri = format!("/v1/files?{query}");
            let (status, body) = terminal_parts(run(&filter(Outcome::Found), Method::GET, &uri, true).await);
            assert_eq!(status, 400);
            assert_eq!(body["error"]["param"], json!("limit"));
        }
    }

    #[tokio::test]
    async fn unprovisioned_catalog_is_503() {
        let (status, body) = terminal_parts(run(&OpenaiFilesFilter::default(), Method::GET, "/v1/files", true).await);
        assert_eq!(status, 503);
        assert_eq!(body["error"]["type"], "server_error");
    }

    #[tokio::test]
    async fn other_requests_continue_to_the_pipeline() {
        let filter = filter(Outcome::Found);
        for (method, uri) in [
            (Method::POST, "/v1/files"),
            (Method::DELETE, "/v1/files/file_1"),
            (Method::GET, "/v1/files/file_1/content"),
            (Method::GET, "/v1/models"),
            (Method::POST, "/v1/responses"),
        ] {
            // Owner-less on purpose: passthrough must not demand one.
            let action = run(&filter, method.clone(), uri, false).await;
            assert!(matches!(action, FilterAction::Continue), "{method} {uri}");
        }
    }

    #[test]
    fn logs_carry_no_filenames_ids_or_error_text() {
        let output = capture_log_output(|| {
            tokio::runtime::Builder::new_current_thread()
                .build()
                .unwrap()
                .block_on(async {
                    for (outcome, uri) in [
                        (Outcome::Found, "/v1/files"),
                        (Outcome::Found, "/v1/files/marker-id"),
                        (Outcome::Missing, "/v1/files/marker-id"),
                        (Outcome::CatalogDown, "/v1/files?after=marker-cursor"),
                    ] {
                        drop(run(&filter(outcome), Method::GET, uri, true).await);
                    }
                });
        });
        assert!(output.contains("operation"), "expected events: {output}");
        for forbidden in ["marker-filename", "marker-id", "marker-cursor", "secret-dsn"] {
            assert!(!output.contains(forbidden), "{forbidden} leaked into logs: {output}");
        }
    }
}
