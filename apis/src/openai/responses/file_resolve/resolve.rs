// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Resolution logic for `file_id` references in Responses API input.
//!
//! Walks the parsed JSON body, finds `file_id` references in
//! `message` content arrays and `function_call_output` output
//! arrays, fetches file metadata and content from the Files API
//! via [`ApiClient`], and replaces the reference with inline
//! content: raw base64 in `file_data` for `input_file`, or a
//! `data:` URL in `image_url` for `input_image`.
//!
//! `file_id` is an `OpenAI` schema extension that is not part of
//! baseline `OpenResponses`. This adapter deliberately accepts that
//! extension and emits the baseline inline `file_data` / `image_url`
//! fields before proxying the request.
//!
//! [`ApiClient`]: crate::openai::api_client::ApiClient

use std::collections::HashMap;

use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use tracing::{debug, warn};

use super::{
    config::OnMissing,
    resolve_url::{FileUrlResolver, redact_url},
};
use crate::openai::api_client::{ApiClient, ApiClientError};

/// Files API path prefix used in resource URL construction.
const FILES_PATH_PREFIX: &str = "v1/files";

/// Identifies the source of a file reference for dispatch and caching.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) enum ReferenceSource {
    /// Files API `file_id` reference.
    FileId(String),
    /// Remote `file_url` reference.
    FileUrl(String),
}

impl std::fmt::Display for ReferenceSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FileId(id) => write!(f, "{id}"),
            Self::FileUrl(url) => write!(f, "{}", redact_url(url)),
        }
    }
}

/// Errors that can occur during file resolution.
#[derive(Debug, Clone)]
pub(crate) enum ResolveError {
    /// The Files API callout failed (non-2xx, timeout, or circuit open).
    CalloutFailed {
        /// The file ID that was requested.
        file_id: String,
        /// Human-readable error description.
        detail: String,
    },

    /// The file ID cannot be represented safely in a Files API path.
    InvalidFileId {
        /// The file ID that was requested.
        file_id: String,
        /// Human-readable error description.
        detail: String,
    },

    /// The request contains more file references than configured.
    TooManyReferences {
        /// Maximum references allowed for one request.
        limit: usize,
    },

    /// Resolved content would exceed the configured body limit.
    TooLarge {
        /// Redacted file reference label.
        reference: String,
        /// Maximum allowed resolved size in bytes.
        limit: usize,
    },

    /// The file URL target is blocked by SSRF policy.
    FileUrlBlocked {
        /// Redacted URL label for client-facing messages.
        label: String,
    },

    /// The file URL fetch failed (DNS, timeout, non-success status).
    FileUrlFailed {
        /// Redacted URL label for client-facing messages.
        label: String,
        /// Human-readable error description.
        detail: String,
    },
}

impl std::fmt::Display for ResolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CalloutFailed { file_id, detail } => {
                write!(f, "callout failed for file '{file_id}': {detail}")
            },
            Self::InvalidFileId { file_id, detail } => {
                write!(f, "invalid file id '{file_id}': {detail}")
            },
            Self::TooManyReferences { limit } => {
                write!(f, "request exceeds configured file reference limit ({limit})")
            },
            Self::TooLarge { reference, limit } => {
                write!(
                    f,
                    "resolved file reference '{reference}' exceeds configured limit ({limit} bytes)"
                )
            },
            Self::FileUrlBlocked { label } => {
                write!(f, "file URL '{label}' blocked by security policy")
            },
            Self::FileUrlFailed { label, detail } => {
                write!(f, "file URL fetch failed for '{label}': {detail}")
            },
        }
    }
}

/// Map an [`ApiClientError`] to a [`ResolveError`], adding the
/// file ID context that the shared client does not carry.
fn map_api_error(err: ApiClientError, file_id: &str) -> ResolveError {
    match err {
        ApiClientError::CalloutFailed { detail } => ResolveError::CalloutFailed {
            file_id: file_id.to_owned(),
            detail,
        },
        ApiClientError::InvalidResourceId { resource_id, detail } => ResolveError::InvalidFileId {
            file_id: resource_id,
            detail,
        },
        ApiClientError::ResponseTooLarge { limit } => ResolveError::TooLarge {
            reference: file_id.to_owned(),
            limit,
        },
        ApiClientError::DecodeFailed { detail } => ResolveError::CalloutFailed {
            file_id: file_id.to_owned(),
            detail: format!("metadata response invalid: {detail}"),
        },
    }
}

/// HTTP client wrapper for fetching files from an OpenAI-compatible
/// Files API, backed by [`ApiClient`] for metadata and bounded
/// byte reads for content.
pub(crate) struct FilesApiClient {
    /// Shared API client providing HTTP operations.
    client: ApiClient,

    /// Maximum number of file references processed per request.
    max_file_references: usize,

    /// Maximum allowed size of one resolved inline data URL.
    max_resolved_bytes: usize,

    /// Overall deadline applied to all resolution callouts for one request.
    resolution_timeout: std::time::Duration,
}

/// Request-scoped limits and cached resolution outcomes.
pub(crate) struct ResolutionBudget {
    /// Resolutions cached by content part type and reference source.
    cache: HashMap<(String, ReferenceSource), Result<ResolvedFile, ResolveError>>,
    /// Deadline shared by every Files API callout in the request.
    deadline: tokio::time::Instant,
    /// Maximum file references allowed for this request.
    max_file_references: usize,
    /// Maximum inline bytes available to each state representation.
    max_resolved_bytes: usize,
    /// File references encountered so far.
    references_seen: usize,
    /// Inline bytes still available across current input and state history.
    remaining_resolved_bytes: usize,
}

/// Count and byte accounting saved while a mirrored state
/// representation is resolved with an independent budget.
#[derive(Clone, Copy)]
pub(crate) struct ResolutionAccounting {
    /// File references charged to the previous representation.
    references_seen: usize,
    /// Inline bytes remaining for the previous representation.
    remaining_resolved_bytes: usize,
}

/// Runtime limits for a Files API client.
#[derive(Clone, Copy)]
pub(crate) struct FilesApiClientOptions {
    /// Maximum distinct file references per request.
    pub max_file_references: usize,
    /// Maximum bytes added by resolved inline content.
    pub max_resolved_bytes: usize,
}

/// Inputs needed to resolve and cache one file reference.
struct ResolutionRequest<'a> {
    /// Files API client.
    client: &'a FilesApiClient,
    /// Reference source (file ID or file URL).
    source: ReferenceSource,
    /// Maximum resolved bytes remaining for this item collection.
    max_resolved_bytes: usize,
    /// Responses content part type.
    part_type: &'a str,
    /// Original request headers available for forwarding.
    request_headers: &'a http::HeaderMap,
    /// URL resolver for `file_url` references.
    url_resolver: Option<&'a FileUrlResolver>,
}

/// Shared dependencies for walking content parts in one item collection.
struct ContentResolver<'a> {
    /// Request-scoped count, deadline, and resolution cache.
    budget: &'a mut ResolutionBudget,
    /// Files API client.
    client: &'a FilesApiClient,
    /// Configured behavior when a file cannot be resolved.
    on_missing: OnMissing,
    /// Original request headers available for forwarding.
    request_headers: &'a http::HeaderMap,
    /// URL resolver for `file_url` references.
    url_resolver: Option<&'a FileUrlResolver>,
}

impl ResolutionBudget {
    /// Start fresh count and byte accounting for a mirrored state
    /// representation while retaining the shared cache and deadline.
    pub(crate) fn begin_independent_accounting(&mut self) -> ResolutionAccounting {
        let saved = ResolutionAccounting {
            references_seen: self.references_seen,
            remaining_resolved_bytes: self.remaining_resolved_bytes,
        };
        self.references_seen = 0;
        self.remaining_resolved_bytes = self.max_resolved_bytes;
        saved
    }

    /// Restore accounting for the authoritative outbound representation.
    pub(crate) fn restore_accounting(&mut self, saved: ResolutionAccounting) {
        self.references_seen = saved.references_seen;
        self.remaining_resolved_bytes = saved.remaining_resolved_bytes;
    }

    /// Resolve one reference within request-wide count and time limits.
    #[expect(clippy::too_many_lines, reason = "inline dispatch logic for FileId vs FileUrl")]
    async fn resolve(&mut self, request: ResolutionRequest<'_>) -> Result<ResolvedFile, ResolveError> {
        let ResolutionRequest {
            client,
            source,
            max_resolved_bytes,
            part_type,
            request_headers,
            url_resolver,
        } = request;
        let key = (part_type.to_owned(), source.clone());
        if let Some(cached) = self.cache.get(&key) {
            return cached.clone();
        }

        self.register_reference()?;

        let resolution = tokio::time::timeout_at(self.deadline, async {
            match &source {
                ReferenceSource::FileId(file_id) => {
                    client
                        .resolve_file(file_id, request_headers, max_resolved_bytes, part_type)
                        .await
                },
                ReferenceSource::FileUrl(url) => {
                    let Some(resolver) = url_resolver else {
                        return Err(ResolveError::FileUrlFailed {
                            label: redact_url(url),
                            detail: "file URL resolution not configured".to_owned(),
                        });
                    };
                    resolver.resolve_url(url, self.deadline, max_resolved_bytes).await
                },
            }
        })
        .await
        .unwrap_or_else(|_elapsed| overall_timeout_error_for_source(&source));

        // Retain one owned outcome for repeated references while
        // returning an independently owned value to the JSON part.
        self.cache.insert(key, resolution.clone());
        resolution
    }

    /// Count one previously unseen reference against the request cap.
    fn register_reference(&mut self) -> Result<(), ResolveError> {
        self.references_seen += 1;
        if self.references_seen > self.max_file_references {
            return Err(ResolveError::TooManyReferences {
                limit: self.max_file_references,
            });
        }
        Ok(())
    }

    /// Consume inline bytes from the request-wide aggregate budget.
    fn consume_resolved_bytes(&mut self, bytes: usize, limit: usize) -> Result<(), ResolveError> {
        self.remaining_resolved_bytes =
            self.remaining_resolved_bytes
                .checked_sub(bytes)
                .ok_or_else(|| ResolveError::TooLarge {
                    reference: "<aggregate>".to_owned(),
                    limit,
                })?;
        Ok(())
    }
}

/// Build the stable error returned when the request-wide deadline expires.
fn overall_timeout_error_for_source(source: &ReferenceSource) -> Result<ResolvedFile, ResolveError> {
    match source {
        ReferenceSource::FileId(file_id) => Err(ResolveError::CalloutFailed {
            file_id: file_id.clone(),
            detail: "overall file resolution deadline exceeded".to_owned(),
        }),
        ReferenceSource::FileUrl(url) => Err(ResolveError::FileUrlFailed {
            label: redact_url(url),
            detail: "overall file resolution deadline exceeded".to_owned(),
        }),
    }
}

impl FilesApiClient {
    /// Build a new client from a pre-built [`ApiClient`] and
    /// domain-specific options.
    pub(crate) fn new(client: ApiClient, options: FilesApiClientOptions) -> Self {
        let FilesApiClientOptions {
            max_file_references,
            max_resolved_bytes,
        } = options;
        Self {
            resolution_timeout: client.timeout(),
            client,
            max_file_references,
            max_resolved_bytes,
        }
    }

    /// Create request-scoped resolution limits and cache state.
    pub(crate) fn resolution_budget(&self) -> ResolutionBudget {
        ResolutionBudget {
            cache: HashMap::new(),
            deadline: tokio::time::Instant::now() + self.resolution_timeout,
            max_file_references: self.max_file_references,
            max_resolved_bytes: self.max_resolved_bytes,
            references_seen: 0,
            remaining_resolved_bytes: self.max_resolved_bytes,
        }
    }

    /// Fetch file metadata from `GET /v1/files/{file_id}`.
    async fn fetch_metadata(
        &self,
        file_id: &str,
        request_headers: &http::HeaderMap,
    ) -> Result<FileMetadata, ResolveError> {
        let url = self
            .client
            .resource_url(FILES_PATH_PREFIX, file_id, None)
            .map_err(|e| map_api_error(e, file_id))?;

        let body = self
            .client
            .get_json(url, request_headers)
            .await
            .map_err(|e| map_api_error(e, file_id))?;

        Ok(parse_file_metadata(&body))
    }

    /// Fetch file content from `GET /v1/files/{file_id}/content`
    /// using bounded reads.
    async fn fetch_content(
        &self,
        file_id: &str,
        request_headers: &http::HeaderMap,
        max_content_bytes: usize,
        max_resolved_bytes: usize,
    ) -> Result<Vec<u8>, ResolveError> {
        let url = self
            .client
            .resource_url(FILES_PATH_PREFIX, file_id, Some("content"))
            .map_err(|e| map_api_error(e, file_id))?;

        self.client
            .get_bytes(&url, request_headers, max_content_bytes)
            .await
            .map_err(|e| match e {
                ApiClientError::ResponseTooLarge { .. } => ResolveError::TooLarge {
                    reference: file_id.to_owned(),
                    limit: max_resolved_bytes,
                },
                other => map_api_error(other, file_id),
            })
    }

    /// Fetch metadata and content, returning the base64 content
    /// and MIME type for the caller to format per the schema.
    async fn resolve_file(
        &self,
        file_id: &str,
        request_headers: &http::HeaderMap,
        max_resolved_bytes: usize,
        part_type: &str,
    ) -> Result<ResolvedFile, ResolveError> {
        let metadata = self.fetch_metadata(file_id, request_headers).await?;
        let max_content_bytes = match part_type {
            "input_image" => max_content_bytes_for_data_url(max_resolved_bytes, &metadata.content_type),
            _ => Some(max_content_bytes_for_base64(max_resolved_bytes)),
        }
        .ok_or_else(|| ResolveError::TooLarge {
            reference: file_id.to_owned(),
            limit: max_resolved_bytes,
        })?;
        if metadata
            .bytes
            .is_some_and(|b| usize::try_from(b).unwrap_or(usize::MAX) > max_content_bytes)
        {
            return Err(ResolveError::TooLarge {
                reference: file_id.to_owned(),
                limit: max_resolved_bytes,
            });
        }
        let content = self
            .fetch_content(file_id, request_headers, max_content_bytes, max_resolved_bytes)
            .await?;
        let base64 = BASE64.encode(&content);

        debug!(file_id, content_type = %metadata.content_type, bytes = content.len(), "resolved file");
        Ok(ResolvedFile {
            base64,
            content_type: metadata.content_type,
            filename: metadata.filename,
        })
    }
}

/// Extract content type and reported size from a Files API
/// metadata response.
fn parse_file_metadata(body: &serde_json::Value) -> FileMetadata {
    let content_type = body
        .get("content_type")
        .and_then(serde_json::Value::as_str)
        .or_else(|| infer_mime_from_filename(body.get("filename").and_then(serde_json::Value::as_str)))
        .unwrap_or("application/octet-stream")
        .to_owned();
    let bytes = body.get("bytes").and_then(serde_json::Value::as_u64);
    let filename = body
        .get("filename")
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned);
    FileMetadata {
        bytes,
        content_type,
        filename,
    }
}

/// File metadata from the Files API.
struct FileMetadata {
    /// Reported file size in bytes (`None` if absent from the
    /// metadata response).
    bytes: Option<u64>,
    /// MIME content type (e.g. `application/pdf`).
    content_type: String,
    /// Original filename (e.g. `report.pdf`).
    filename: Option<String>,
}

/// Resolved file content ready for inlining.
#[derive(Clone)]
pub(crate) struct ResolvedFile {
    /// Base64-encoded file content (no `data:` prefix).
    pub(super) base64: String,
    /// MIME content type (e.g. `text/plain`).
    pub(super) content_type: String,
    /// Original filename from the Files API metadata.
    pub(super) filename: Option<String>,
}

/// Walk the Responses API `input` array and resolve all references in place.
///
/// Returns the number of references resolved.
#[cfg(test)]
pub(crate) async fn resolve_input(
    body: &mut serde_json::Value,
    client: &FilesApiClient,
    on_missing: OnMissing,
    request_headers: &http::HeaderMap,
    url_resolver: Option<&FileUrlResolver>,
) -> Result<usize, ResolveError> {
    let mut budget = client.resolution_budget();
    resolve_input_with_budget(body, client, on_missing, request_headers, url_resolver, &mut budget).await
}

/// Resolve request input using limits shared with other request state.
#[expect(clippy::too_many_arguments, reason = "threading url_resolver through resolution")]
pub(crate) async fn resolve_input_with_budget(
    body: &mut serde_json::Value,
    client: &FilesApiClient,
    on_missing: OnMissing,
    request_headers: &http::HeaderMap,
    url_resolver: Option<&FileUrlResolver>,
    budget: &mut ResolutionBudget,
) -> Result<usize, ResolveError> {
    let Some(serde_json::Value::Array(input)) = body.get_mut("input") else {
        return Ok(0);
    };

    resolve_items(input, client, on_missing, request_headers, url_resolver, budget).await
}

/// Walk a slice of input/message items and resolve all file references
/// in their content arrays.
///
/// Handles `message` items (with `content`), `function_call_output`
/// items (with `output`), and the shorthand message form.
#[expect(clippy::too_many_arguments, reason = "threading url_resolver through resolution")]
pub(crate) async fn resolve_items(
    items: &mut [serde_json::Value],
    client: &FilesApiClient,
    on_missing: OnMissing,
    request_headers: &http::HeaderMap,
    url_resolver: Option<&FileUrlResolver>,
    budget: &mut ResolutionBudget,
) -> Result<usize, ResolveError> {
    let mut resolved_count = 0_usize;
    let mut resolver = ContentResolver {
        budget,
        client,
        on_missing,
        request_headers,
        url_resolver,
    };

    for item in items.iter_mut() {
        resolved_count += resolve_item(item, &mut resolver).await?;
    }

    Ok(resolved_count)
}

/// Resolve every supported content part in one input item.
async fn resolve_item(item: &mut serde_json::Value, resolver: &mut ContentResolver<'_>) -> Result<usize, ResolveError> {
    let Some(parts) = content_parts_mut(item) else {
        return Ok(0);
    };
    let mut resolved_count = 0_usize;
    for part in parts {
        if let Some(resolved_len) = resolve_content_part(part, resolver).await? {
            resolver
                .budget
                .consume_resolved_bytes(resolved_len, resolver.client.max_resolved_bytes)?;
            resolved_count += 1;
        }
    }
    Ok(resolved_count)
}

/// Return the mutable content parts array for a given input item,
/// if applicable.
pub(crate) fn content_parts_mut(item: &mut serde_json::Value) -> Option<&mut Vec<serde_json::Value>> {
    match item.get("type").and_then(serde_json::Value::as_str) {
        Some("message") => item.get_mut("content").and_then(serde_json::Value::as_array_mut),
        Some("function_call_output") => item.get_mut("output").and_then(serde_json::Value::as_array_mut),
        Some(_) => None,
        None => {
            if item.get("role").and_then(serde_json::Value::as_str).is_some() && item.get("content").is_some() {
                item.get_mut("content").and_then(serde_json::Value::as_array_mut)
            } else {
                None
            }
        },
    }
}

/// Resolve a single content part if it contains a resolvable reference.
async fn resolve_content_part(
    part: &mut serde_json::Value,
    resolver: &mut ContentResolver<'_>,
) -> Result<Option<usize>, ResolveError> {
    let Some((part_type, source)) = resolvable_reference(part) else {
        return Ok(None);
    };

    // Skip file_url references when url_resolver is not configured.
    if matches!(source, ReferenceSource::FileUrl(_)) && resolver.url_resolver.is_none() {
        return Ok(None);
    }

    let (source, part_type) = (source, part_type.to_owned());
    debug!(source = %source, part_type = %part_type, "resolving file reference");

    let max_resolved_bytes = resolver.budget.remaining_resolved_bytes;
    let Some(resolved) = resolve_reference(&source, &part_type, max_resolved_bytes, resolver).await? else {
        return Ok(None);
    };
    let len = output_len_for_part(&part_type, &source, &resolved);
    if len > max_resolved_bytes {
        return Err(ResolveError::TooLarge {
            reference: source.to_string(),
            limit: resolver.client.max_resolved_bytes,
        });
    }
    rewrite_part(part, &part_type, &source, resolved);
    Ok(Some(len))
}

/// Resolve one reference and apply the configured missing-file policy.
async fn resolve_reference(
    source: &ReferenceSource,
    part_type: &str,
    remaining_resolved_bytes: usize,
    resolver: &mut ContentResolver<'_>,
) -> Result<Option<ResolvedFile>, ResolveError> {
    match resolver
        .budget
        .resolve(ResolutionRequest {
            client: resolver.client,
            source: source.clone(),
            max_resolved_bytes: remaining_resolved_bytes,
            part_type,
            request_headers: resolver.request_headers,
            url_resolver: resolver.url_resolver,
        })
        .await
    {
        Ok(resolved) => Ok(Some(resolved)),
        Err(e @ (ResolveError::TooManyReferences { .. } | ResolveError::FileUrlBlocked { .. })) => Err(e),
        Err(e) if resolver.on_missing == OnMissing::Continue => {
            warn!(source = %source, error = %e, "file resolution failed, passing through");
            Ok(None)
        },
        Err(e) => Err(e),
    }
}

/// Return the type and reference source for a part that needs resolution.
///
/// Classifies `input_file` parts by their single valid source field:
/// `file_id` or `file_url`. Parts with multiple sources, no sources, or
/// malformed (present, non-null, non-string) fields are skipped. Null
/// values are treated as absent per the Responses schema.
///
/// `input_image` parts continue to resolve only via `file_id`.
#[expect(clippy::too_many_lines, reason = "explicit field validation logic")]
fn resolvable_reference(part: &serde_json::Value) -> Option<(&str, ReferenceSource)> {
    let part_type @ "input_file" = part.get("type")?.as_str()? else {
        // input_image file_id resolution is unchanged — keep the
        // existing has_inline_content/file_id path for it.
        if part.get("type")?.as_str()? == "input_image" {
            return resolvable_image_reference(part);
        }
        return None;
    };

    // Count valid string sources and detect malformed (present, non-null,
    // non-string) fields. Null is valid per the Responses schema and
    // treated as absent.
    let mut valid_sources = 0_u8;
    let mut has_malformed = false;
    let mut file_id: Option<&str> = None;
    let mut file_url_str: Option<&str> = None;

    for field in ["file_data", "file_id", "file_url"] {
        if let Some(val) = part.get(field) {
            if val.is_null() {
                continue;
            }
            if let Some(s) = val.as_str() {
                valid_sources += 1;
                match field {
                    "file_id" => file_id = Some(s),
                    "file_url" => file_url_str = Some(s),
                    _ => {},
                }
            } else {
                has_malformed = true;
            }
        }
    }

    if has_malformed || valid_sources != 1 {
        return None;
    }

    if let Some(url) = file_url_str {
        return Some((part_type, ReferenceSource::FileUrl(url.to_owned())));
    }
    if let Some(id) = file_id {
        return Some((part_type, ReferenceSource::FileId(id.to_owned())));
    }
    None
}

/// Existing `input_image` `file_id` resolution (unchanged behavior).
fn resolvable_image_reference(part: &serde_json::Value) -> Option<(&'static str, ReferenceSource)> {
    if part.get("image_url").and_then(serde_json::Value::as_str).is_some() {
        return None;
    }
    let file_id = part.get("file_id")?.as_str()?;
    Some(("input_image", ReferenceSource::FileId(file_id.to_owned())))
}

/// Compute the JSON string length of the resolved value for a
/// given content part type and source.
fn output_len_for_part(part_type: &str, source: &ReferenceSource, resolved: &ResolvedFile) -> usize {
    match (part_type, source) {
        ("input_file", ReferenceSource::FileId(_)) => resolved.base64.len(),
        ("input_file", ReferenceSource::FileUrl(_)) => {
            "data:".len() + resolved.content_type.len() + ";base64,".len() + resolved.base64.len()
        },
        ("input_image", _) => "data:".len() + resolved.content_type.len() + ";base64,".len() + resolved.base64.len(),
        _ => 0,
    }
}

/// Replace the reference field with the resolved content in a content part.
///
/// For `input_file` with `FileId`, writes raw base64 to `file_data`.
/// For `input_file` with `FileUrl`, writes a data URI to `file_data`.
/// For `input_image`, writes a `data:` URL to `image_url`.
/// Populates `filename` from metadata when not already user-provided.
#[expect(clippy::too_many_lines, reason = "explicit branching per part type and source")]
fn rewrite_part(part: &mut serde_json::Value, part_type: &str, source: &ReferenceSource, resolved: ResolvedFile) {
    let Some(obj) = part.as_object_mut() else {
        return;
    };
    let ResolvedFile {
        mut base64,
        content_type,
        filename,
    } = resolved;

    match source {
        ReferenceSource::FileId(_) => {
            obj.remove("file_id");
        },
        ReferenceSource::FileUrl(_) => {
            obj.remove("file_url");
        },
    }

    match (part_type, source) {
        ("input_file", ReferenceSource::FileId(_)) => {
            obj.insert("file_data".to_owned(), serde_json::Value::String(base64));
            if !obj.contains_key("filename")
                && let Some(filename) = filename
            {
                obj.insert("filename".to_owned(), serde_json::Value::String(filename));
            }
        },
        ("input_file", ReferenceSource::FileUrl(_)) => {
            let data_uri = format!("data:{content_type};base64,{base64}");
            obj.insert("file_data".to_owned(), serde_json::Value::String(data_uri));
            if !obj.contains_key("filename")
                && let Some(filename) = filename
            {
                obj.insert("filename".to_owned(), serde_json::Value::String(filename));
            }
        },
        ("input_image", _) => {
            let prefix = format!("data:{content_type};base64,");
            base64.insert_str(0, &prefix);
            obj.insert("image_url".to_owned(), serde_json::Value::String(base64));
        },
        _ => {},
    }
}

/// Maximum raw file bytes whose base64 encoding fits in
/// `max_output_bytes`.
fn max_content_bytes_for_base64(max_output_bytes: usize) -> usize {
    (max_output_bytes / 4) * 3
}

/// Maximum raw file bytes that can fit in a data URL with base64
/// expansion.
pub(super) fn max_content_bytes_for_data_url(max_data_url_bytes: usize, content_type: &str) -> Option<usize> {
    let prefix_len = "data:"
        .len()
        .checked_add(content_type.len())?
        .checked_add(";base64,".len())?;
    let available = max_data_url_bytes.checked_sub(prefix_len)?;
    Some((available / 4) * 3)
}

/// Infer MIME type from a filename extension.
pub(crate) fn infer_mime_from_filename(filename: Option<&str>) -> Option<&'static str> {
    let ext = filename?.rsplit('.').next()?;
    match ext.to_ascii_lowercase().as_str() {
        "csv" => Some("text/csv"),
        "doc" => Some("application/msword"),
        "docx" => Some("application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
        "gif" => Some("image/gif"),
        "html" | "htm" => Some("text/html"),
        "jpg" | "jpeg" => Some("image/jpeg"),
        "json" => Some("application/json"),
        "pdf" => Some("application/pdf"),
        "png" => Some("image/png"),
        "pptx" => Some("application/vnd.openxmlformats-officedocument.presentationml.presentation"),
        "txt" => Some("text/plain"),
        "webp" => Some("image/webp"),
        "xlsx" => Some("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
        "xml" => Some("application/xml"),
        _ => None,
    }
}

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    reason = "tests"
)]
mod tests {
    use std::{
        io::{Read as _, Write as _},
        net::TcpListener,
    };

    use praxis_core::callout::{CalloutConfig, FailureMode};

    use super::*;
    use crate::openai::api_client::{ApiClient, ApiClientConfig};

    #[test]
    fn infer_mime_pdf() {
        assert_eq!(
            infer_mime_from_filename(Some("document.pdf")),
            Some("application/pdf"),
            "PDF extension should map to application/pdf"
        );
    }

    #[test]
    fn infer_mime_png() {
        assert_eq!(
            infer_mime_from_filename(Some("image.png")),
            Some("image/png"),
            "PNG extension should map to image/png"
        );
    }

    #[test]
    fn infer_mime_case_insensitive() {
        assert_eq!(
            infer_mime_from_filename(Some("IMAGE.PNG")),
            Some("image/png"),
            "extension matching should be case-insensitive"
        );
    }

    #[test]
    fn infer_mime_unknown_extension() {
        assert_eq!(
            infer_mime_from_filename(Some("archive.tar.gz")),
            None,
            "unknown extension should return None"
        );
    }

    #[test]
    fn infer_mime_no_filename() {
        assert_eq!(infer_mime_from_filename(None), None, "None filename should return None");
    }

    #[test]
    fn resolve_error_display_callout_failed() {
        let err = ResolveError::CalloutFailed {
            file_id: "file-abc".to_owned(),
            detail: "connection refused".to_owned(),
        };
        assert_eq!(
            err.to_string(),
            "callout failed for file 'file-abc': connection refused",
            "CalloutFailed display should format correctly"
        );
    }

    #[test]
    fn resolve_error_display_too_large() {
        let err = ResolveError::TooLarge {
            reference: "file-abc".to_owned(),
            limit: 1024,
        };
        assert_eq!(
            err.to_string(),
            "resolved file reference 'file-abc' exceeds configured limit (1024 bytes)",
            "TooLarge display should format correctly"
        );
    }

    #[test]
    fn resolve_error_display_too_many_references() {
        let err = ResolveError::TooManyReferences { limit: 32 };

        assert_eq!(
            err.to_string(),
            "request exceeds configured file reference limit (32)",
            "TooManyReferences display should include the configured limit"
        );
    }

    #[test]
    fn max_content_bytes_accounts_for_base64_expansion() {
        let prefix_len = "data:text/plain;base64,".len();

        assert_eq!(
            max_content_bytes_for_data_url(prefix_len + 4, "text/plain"),
            Some(3),
            "four base64 bytes can carry three raw bytes"
        );
        assert_eq!(
            max_content_bytes_for_data_url(prefix_len + 3, "text/plain"),
            Some(0),
            "partial base64 quantum should not allow extra raw bytes"
        );
        assert_eq!(
            max_content_bytes_for_data_url(prefix_len - 1, "text/plain"),
            None,
            "limit smaller than data URL prefix should reject before fetch"
        );
    }

    #[test]
    fn parse_metadata_extracts_bytes_and_filename() {
        let body = serde_json::json!({
            "id": "file-abc",
            "content_type": "text/plain",
            "filename": "test.txt",
            "bytes": 12345
        });
        let meta = parse_file_metadata(&body);
        assert_eq!(meta.bytes, Some(12345), "bytes should be extracted from metadata");
        assert_eq!(meta.content_type, "text/plain");
        assert_eq!(
            meta.filename.as_deref(),
            Some("test.txt"),
            "filename should be extracted"
        );
    }

    #[test]
    fn parse_metadata_missing_bytes_returns_none() {
        let body = serde_json::json!({
            "id": "file-abc",
            "content_type": "text/plain",
            "filename": "test.txt"
        });
        let meta = parse_file_metadata(&body);
        assert_eq!(meta.bytes, None, "bytes should be None when absent from metadata");
    }

    fn test_api_client(api_base_url: &str, timeout_ms: u64) -> ApiClient {
        ApiClient::new(ApiClientConfig {
            api_base_url: api_base_url.to_owned(),
            callout_config: CalloutConfig {
                failure_mode: FailureMode::Closed,
                timeout_ms,
                ..CalloutConfig::default()
            },
            forward_header_names: Vec::new(),
        })
        .unwrap()
    }

    fn test_client(api_base_url: &str) -> FilesApiClient {
        test_client_with_limits(api_base_url, 1024, 1_000)
    }

    fn test_client_with_limits(api_base_url: &str, max_resolved_bytes: usize, timeout_ms: u64) -> FilesApiClient {
        let api = test_api_client(api_base_url, timeout_ms);
        FilesApiClient::new(
            api,
            FilesApiClientOptions {
                max_file_references: 32,
                max_resolved_bytes,
            },
        )
    }

    #[tokio::test]
    async fn content_download_does_not_follow_redirects() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = [0_u8; 4096];
            let _read = stream.read(&mut request).unwrap();
            stream
                .write_all(
                    b"HTTP/1.1 302 Found\r\nLocation: http://127.0.0.1:9/secret\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                )
                .unwrap();
        });
        let client = test_client(&format!("http://{address}"));

        let err = client
            .fetch_content("file-redirect", &http::HeaderMap::new(), 1024, 1024)
            .await
            .unwrap_err();

        assert!(
            matches!(err, ResolveError::CalloutFailed { .. }),
            "redirect response should be rejected without contacting its target: {err}"
        );
    }

    #[tokio::test]
    async fn content_download_transport_failure_sanitizes_error_detail() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            drop(stream);
        });
        let client = test_client(&format!("http://{address}"));

        let err = client
            .fetch_content("file-disconnect", &http::HeaderMap::new(), 1024, 1024)
            .await
            .unwrap_err();

        assert!(
            matches!(err, ResolveError::CalloutFailed { .. }),
            "transport errors should map to CalloutFailed: {err}"
        );
    }

    #[tokio::test]
    async fn content_download_without_length_is_bounded_while_streaming() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = [0_u8; 4096];
            let _read = stream.read(&mut request).unwrap();
            stream
                .write_all(b"HTTP/1.1 200 OK\r\nConnection: close\r\n\r\n0123456789abcdef")
                .unwrap();
        });
        let client = test_client(&format!("http://{address}"));

        let err = client
            .fetch_content("file-large", &http::HeaderMap::new(), 8, 1024)
            .await
            .unwrap_err();

        assert!(
            matches!(err, ResolveError::TooLarge { .. }),
            "streamed bodies without Content-Length should be rejected at the byte limit"
        );
    }

    #[tokio::test]
    async fn resolve_input_accepts_message_shorthand_without_type() {
        let client = test_client("http://files-api:8321");
        let mut body = serde_json::json!({
            "input": [{
                "role": "user",
                "content": [{"type": "input_file", "file_id": ".."}]
            }]
        });
        let err = resolve_input(&mut body, &client, OnMissing::Reject, &http::HeaderMap::new(), None)
            .await
            .unwrap_err();

        assert!(
            matches!(err, ResolveError::InvalidFileId { .. }),
            "shorthand message items should be walked and attempt file resolution"
        );
    }

    #[tokio::test]
    async fn resolve_input_walks_function_call_output_arrays() {
        let client = test_client("http://files-api:8321");
        let mut body = serde_json::json!({
            "input": [{
                "type": "function_call_output",
                "call_id": "call-123",
                "output": [{"type": "input_file", "file_id": ".."}]
            }]
        });

        let err = resolve_input(&mut body, &client, OnMissing::Reject, &http::HeaderMap::new(), None)
            .await
            .unwrap_err();

        assert!(
            matches!(err, ResolveError::InvalidFileId { .. }),
            "function call output arrays should be walked for file references"
        );
    }

    #[tokio::test]
    async fn continue_policy_preserves_unresolvable_reference() {
        let client = test_client("http://files-api:8321");
        let mut body = serde_json::json!({
            "input": [{
                "type": "message",
                "role": "user",
                "content": [{"type": "input_file", "file_id": ".."}]
            }]
        });
        let original = body.clone();

        let resolved = resolve_input(&mut body, &client, OnMissing::Continue, &http::HeaderMap::new(), None)
            .await
            .unwrap();

        assert_eq!(resolved, 0, "failed resolution should not increment the count");
        assert_eq!(
            body, original,
            "continue policy should preserve the original file reference"
        );
    }

    #[tokio::test]
    async fn distinct_reference_limit_is_enforced_in_continue_mode() {
        let api = test_api_client("http://files-api:8321", 1_000);
        let mut client = FilesApiClient::new(
            api,
            FilesApiClientOptions {
                max_file_references: 2,
                max_resolved_bytes: 1024,
            },
        );
        client.max_file_references = 2;
        let mut body = serde_json::json!({
            "input": [{
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_file", "file_id": "."},
                    {"type": "input_file", "file_id": ".."},
                    {"type": "input_image", "file_id": "."}
                ]
            }]
        });

        let err = resolve_input(&mut body, &client, OnMissing::Continue, &http::HeaderMap::new(), None)
            .await
            .unwrap_err();

        assert!(
            matches!(err, ResolveError::TooManyReferences { limit: 2 }),
            "reference limit should reject even when missing files normally pass through"
        );
    }

    #[tokio::test]
    async fn repeated_reference_reuses_cached_failure() {
        let api = test_api_client("http://files-api:8321", 1_000);
        let client = FilesApiClient::new(
            api,
            FilesApiClientOptions {
                max_file_references: 1,
                max_resolved_bytes: 1024,
            },
        );
        let mut body = serde_json::json!({
            "input": [{
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_file", "file_id": "."},
                    {"type": "input_file", "file_id": "."}
                ]
            }]
        });

        let count = resolve_input(&mut body, &client, OnMissing::Continue, &http::HeaderMap::new(), None)
            .await
            .unwrap();

        assert_eq!(count, 0, "cached failures should remain unresolved in continue mode");
    }

    #[tokio::test]
    async fn cached_successes_share_request_wide_byte_budget() {
        let client = test_client_with_limits("http://files-api:8321", 8, 1_000);
        let mut budget = client.resolution_budget();
        budget.cache.insert(
            (
                "input_file".to_owned(),
                ReferenceSource::FileId("file-cached".to_owned()),
            ),
            Ok(ResolvedFile {
                base64: "ZGF0".to_owned(),
                content_type: "text/plain".to_owned(),
                filename: Some("data.txt".to_owned()),
            }),
        );

        for _ in 0..2 {
            let count = resolve_cached_items(&client, &mut budget).await.unwrap();
            assert_eq!(
                count, 1,
                "cached content should resolve while aggregate capacity remains"
            );
        }

        let err = resolve_cached_items(&client, &mut budget).await.unwrap_err();
        assert!(
            matches!(err, ResolveError::TooLarge { .. }),
            "cached copies across item collections should share one aggregate byte budget"
        );
    }

    async fn resolve_cached_items(
        client: &FilesApiClient,
        budget: &mut ResolutionBudget,
    ) -> Result<usize, ResolveError> {
        let mut items = cached_file_items();
        resolve_items(
            &mut items,
            client,
            OnMissing::Reject,
            &http::HeaderMap::new(),
            None,
            budget,
        )
        .await
    }

    fn cached_file_items() -> Vec<serde_json::Value> {
        vec![serde_json::json!({
            "type": "message",
            "role": "user",
            "content": [{"type": "input_file", "file_id": "file-cached"}]
        })]
    }

    #[tokio::test]
    async fn inline_content_takes_precedence_over_file_id() {
        let client = test_client("http://files-api:8321");

        for (part_type, field, value) in [
            ("input_file", "file_data", "SGVsbG8="),
            ("input_file", "file_url", "https://example.com/report.pdf"),
            ("input_image", "image_url", "data:image/png;base64,iVBOR"),
        ] {
            let mut inline_part = serde_json::json!({"type": part_type, "file_id": ".."});
            inline_part[field] = serde_json::Value::String(value.to_owned());
            let mut body = serde_json::json!({
                "input": [{
                    "type": "message",
                    "role": "user",
                    "content": [inline_part]
                }]
            });
            let original = body.clone();

            let resolved = resolve_input(&mut body, &client, OnMissing::Reject, &http::HeaderMap::new(), None)
                .await
                .unwrap();

            assert_eq!(resolved, 0, "inline content should not be resolved again");
            assert_eq!(
                body, original,
                "inline content should pass through unchanged even when file_id is also present"
            );
        }
    }

    #[tokio::test]
    async fn resolve_input_skips_untyped_non_message_content() {
        let client = test_client("http://files-api:8321");
        let mut body = serde_json::json!({
            "input": [{
                "content": [{"type": "input_file", "file_id": ".."}]
            }]
        });
        let resolved = resolve_input(&mut body, &client, OnMissing::Reject, &http::HeaderMap::new(), None)
            .await
            .unwrap();

        assert_eq!(
            resolved, 0,
            "untyped items without a role should not be treated as messages"
        );
    }

    #[tokio::test]
    async fn file_data_parts_pass_through_unchanged() {
        let client = test_client("http://files-api:8321");
        let mut body = serde_json::json!({
            "input": [{
                "type": "message",
                "role": "user",
                "content": [{
                    "type": "input_file",
                    "file_data": "SGVsbG8=",
                    "filename": "hello.txt"
                }]
            }]
        });
        let original = body.clone();
        let count = resolve_input(&mut body, &client, OnMissing::Reject, &http::HeaderMap::new(), None)
            .await
            .unwrap();

        assert_eq!(count, 0, "parts with file_data (no file_id) should not be resolved");
        assert_eq!(body, original, "body should be unchanged");
    }

    #[tokio::test]
    async fn file_url_parts_pass_through_unchanged() {
        let client = test_client("http://files-api:8321");
        let mut body = serde_json::json!({
            "input": [{
                "type": "message",
                "role": "user",
                "content": [{
                    "type": "input_file",
                    "file_url": "https://example.com/file.pdf",
                    "filename": "file.pdf"
                }]
            }]
        });
        let original = body.clone();
        let count = resolve_input(&mut body, &client, OnMissing::Reject, &http::HeaderMap::new(), None)
            .await
            .unwrap();

        assert_eq!(count, 0, "parts with file_url (no file_id) should not be resolved");
        assert_eq!(body, original, "body should be unchanged");
    }

    #[tokio::test]
    async fn image_url_parts_pass_through_unchanged() {
        let client = test_client("http://files-api:8321");
        let mut body = serde_json::json!({
            "input": [{
                "type": "message",
                "role": "user",
                "content": [{
                    "type": "input_image",
                    "image_url": "data:image/png;base64,iVBOR",
                    "detail": "high"
                }]
            }]
        });
        let original = body.clone();
        let count = resolve_input(&mut body, &client, OnMissing::Reject, &http::HeaderMap::new(), None)
            .await
            .unwrap();

        assert_eq!(count, 0, "parts with image_url (no file_id) should not be resolved");
        assert_eq!(body, original, "body should be unchanged, including detail field");
    }

    #[test]
    fn rewrite_part_preserves_detail_on_input_image() {
        let mut part = serde_json::json!({
            "type": "input_image",
            "file_id": "img-456",
            "detail": "high"
        });
        let resolved = ResolvedFile {
            base64: "iVBOR".to_owned(),
            content_type: "image/png".to_owned(),
            filename: None,
        };
        rewrite_part(
            &mut part,
            "input_image",
            &ReferenceSource::FileId("img-456".to_owned()),
            resolved,
        );

        assert!(part.get("file_id").is_none(), "file_id should be removed");
        assert_eq!(part["detail"].as_str().unwrap(), "high", "detail should be preserved");
        assert!(
            part["image_url"]
                .as_str()
                .unwrap()
                .starts_with("data:image/png;base64,"),
            "image_url should be set"
        );
    }

    #[test]
    fn rewrite_part_preserves_user_filename_on_input_file() {
        let mut part = serde_json::json!({
            "type": "input_file",
            "file_id": "file-abc",
            "filename": "user-provided.pdf"
        });
        let resolved = ResolvedFile {
            base64: "JVBER".to_owned(),
            content_type: "application/pdf".to_owned(),
            filename: Some("api-filename.pdf".to_owned()),
        };
        rewrite_part(
            &mut part,
            "input_file",
            &ReferenceSource::FileId("file-abc".to_owned()),
            resolved,
        );

        assert_eq!(
            part["filename"].as_str().unwrap(),
            "user-provided.pdf",
            "user-provided filename should take precedence over metadata"
        );
    }

    #[test]
    fn rewrite_part_populates_filename_from_metadata() {
        let mut part = serde_json::json!({
            "type": "input_file",
            "file_id": "file-abc"
        });
        let resolved = ResolvedFile {
            base64: "SGVsbG8=".to_owned(),
            content_type: "text/plain".to_owned(),
            filename: Some("test.txt".to_owned()),
        };
        rewrite_part(
            &mut part,
            "input_file",
            &ReferenceSource::FileId("file-abc".to_owned()),
            resolved,
        );

        assert_eq!(
            part["filename"].as_str().unwrap(),
            "test.txt",
            "filename should be populated from metadata when not provided by user"
        );
    }

    #[test]
    fn reference_source_file_id_and_file_url_are_distinct_cache_keys() {
        use std::collections::HashMap;

        let mut cache = HashMap::new();
        let key_id = ("input_file".to_owned(), ReferenceSource::FileId("abc".to_owned()));
        let key_url = ("input_file".to_owned(), ReferenceSource::FileUrl("abc".to_owned()));
        cache.insert(key_id.clone(), "from_id");
        cache.insert(key_url.clone(), "from_url");

        assert_eq!(cache[&key_id], "from_id", "FileId key should be distinct from FileUrl");
        assert_eq!(
            cache[&key_url], "from_url",
            "FileUrl key should be distinct from FileId"
        );
        assert_eq!(cache.len(), 2, "two distinct keys should produce two entries");
    }

    // -------------------------------------------------------------------------
    // Source classification tests
    // -------------------------------------------------------------------------

    #[test]
    fn resolvable_reference_file_url_only() {
        let part = serde_json::json!({"type": "input_file", "file_url": "https://example.com/file.pdf"});
        let result = resolvable_reference(&part);
        assert!(
            matches!(result, Some(("input_file", ReferenceSource::FileUrl(url))) if url == "https://example.com/file.pdf"),
            "file_url-only part should be classified as FileUrl"
        );
    }

    #[test]
    fn resolvable_reference_file_id_only() {
        let part = serde_json::json!({"type": "input_file", "file_id": "file-abc"});
        let result = resolvable_reference(&part);
        assert!(
            matches!(result, Some(("input_file", ReferenceSource::FileId(id))) if id == "file-abc"),
            "file_id-only part should be classified as FileId"
        );
    }

    #[test]
    fn resolvable_reference_multi_source_skipped() {
        let part = serde_json::json!({"type": "input_file", "file_id": "abc", "file_url": "https://example.com/f"});
        assert!(
            resolvable_reference(&part).is_none(),
            "multi-source parts should be skipped"
        );
    }

    #[test]
    fn resolvable_reference_file_data_skipped() {
        let part = serde_json::json!({"type": "input_file", "file_data": "SGVsbG8="});
        assert!(
            resolvable_reference(&part).is_none(),
            "file_data-only parts should be skipped"
        );
    }

    #[test]
    fn resolvable_reference_malformed_non_string_file_url_skipped() {
        let part = serde_json::json!({"type": "input_file", "file_url": 123, "file_id": "valid-id"});
        assert!(
            resolvable_reference(&part).is_none(),
            "non-string file_url should mark part malformed, skipping even valid file_id"
        );
    }

    #[test]
    fn resolvable_reference_null_file_url_treated_as_absent() {
        let part = serde_json::json!({"type": "input_file", "file_url": null, "file_id": "valid-id"});
        let result = resolvable_reference(&part);
        assert!(
            matches!(result, Some(("input_file", ReferenceSource::FileId(id))) if id == "valid-id"),
            "null file_url should be treated as absent, resolving file_id normally"
        );
    }

    #[test]
    fn resolvable_reference_non_string_file_data_marks_malformed() {
        let part = serde_json::json!({"type": "input_file", "file_data": 42});
        assert!(
            resolvable_reference(&part).is_none(),
            "non-string file_data should mark part malformed"
        );
    }
}
