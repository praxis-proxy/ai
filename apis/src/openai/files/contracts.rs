// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Wire contracts for the OpenAI Files API.
//!
//! These are wire types only: the Files service domain model maps to and from
//! them. Component names match the pinned OpenAI specification so drift checks
//! compare like with like. Deliberate deviations from the pinned reference:
//!
//! - `OpenAIFile.purpose` accepts `evals`, so an upload round-trips.
//! - `downloadFile` declares `application/octet-stream` rather than upstream's `application/json` string, because
//!   Praxis streams the stored bytes.

#![expect(
    clippy::large_stack_frames,
    reason = "utoipa macro-generated schema builders allocate large temporary values"
)]
#![expect(
    dead_code,
    reason = "wire types are consumed by handlers and the multipart parser added in follow-up changes"
)]

use std::{error::Error, fmt, str::FromStr};

use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;
use utoipa::{
    ToSchema,
    openapi::{
        Deprecated, RefOr,
        schema::{Object, ObjectBuilder, Schema, Type},
    },
};

/// Lowest accepted `expires_after.seconds` (one hour).
pub(crate) const MIN_EXPIRATION_SECONDS: u32 = 3_600;

/// Highest accepted `expires_after.seconds` (thirty days).
pub(crate) const MAX_EXPIRATION_SECONDS: u32 = 2_592_000;

/// Lowest accepted list `limit`.
pub(crate) const MIN_LIST_LIMIT: u32 = 1;

/// Highest accepted list `limit`, also its default.
pub(crate) const MAX_LIST_LIMIT: u32 = 10_000;

/// A request field that violates the Files contract.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct InvalidField {
    /// Offending field name.
    pub(crate) field: &'static str,
    /// Why the value was rejected.
    pub(crate) reason: String,
}

impl InvalidField {
    /// Build a rejection for one field.
    fn new(field: &'static str, reason: impl Into<String>) -> Self {
        Self {
            field,
            reason: reason.into(),
        }
    }
}

impl fmt::Display for InvalidField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid `{}`: {}", self.field, self.reason)
    }
}

impl Error for InvalidField {}

// -----------------------------------------------------------------------------
// Enums
// -----------------------------------------------------------------------------

/// Purposes accepted when uploading a file.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, ToSchema)]
pub(crate) enum UploadPurpose {
    /// Used by the Assistants API.
    #[serde(rename = "assistants")]
    Assistants,
    /// Used by the Batch API.
    #[serde(rename = "batch")]
    Batch,
    /// Used for fine-tuning.
    #[serde(rename = "fine-tune")]
    FineTune,
    /// Used for image inputs.
    #[serde(rename = "vision")]
    Vision,
    /// Flexible file type for any purpose.
    #[serde(rename = "user_data")]
    UserData,
    /// Used for eval data sets.
    #[serde(rename = "evals")]
    Evals,
}

impl UploadPurpose {
    /// Wire value.
    #[must_use]
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Assistants => "assistants",
            Self::Batch => "batch",
            Self::FineTune => "fine-tune",
            Self::Vision => "vision",
            Self::UserData => "user_data",
            Self::Evals => "evals",
        }
    }
}

impl FromStr for UploadPurpose {
    type Err = InvalidField;

    /// Parse a multipart text field. Matching is exact and case-sensitive.
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "assistants" => Ok(Self::Assistants),
            "batch" => Ok(Self::Batch),
            "fine-tune" => Ok(Self::FineTune),
            "vision" => Ok(Self::Vision),
            "user_data" => Ok(Self::UserData),
            "evals" => Ok(Self::Evals),
            other => Err(InvalidField::new("purpose", format!("unsupported purpose `{other}`"))),
        }
    }
}

/// Purposes reported on stored files: the upload values plus output purposes.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize, ToSchema)]
pub(crate) enum FilePurpose {
    /// Used by the Assistants API.
    #[serde(rename = "assistants")]
    Assistants,
    /// Output of the Assistants API.
    #[serde(rename = "assistants_output")]
    AssistantsOutput,
    /// Used by the Batch API.
    #[serde(rename = "batch")]
    Batch,
    /// Output of the Batch API.
    #[serde(rename = "batch_output")]
    BatchOutput,
    /// Used for fine-tuning.
    #[serde(rename = "fine-tune")]
    FineTune,
    /// Fine-tuning results.
    #[serde(rename = "fine-tune-results")]
    FineTuneResults,
    /// Used for image inputs.
    #[serde(rename = "vision")]
    Vision,
    /// Flexible file type for any purpose.
    #[serde(rename = "user_data")]
    UserData,
    /// Used for eval data sets. Not in the pinned `OpenAIFile.purpose` enum.
    #[serde(rename = "evals")]
    Evals,
}

impl From<UploadPurpose> for FilePurpose {
    fn from(purpose: UploadPurpose) -> Self {
        match purpose {
            UploadPurpose::Assistants => Self::Assistants,
            UploadPurpose::Batch => Self::Batch,
            UploadPurpose::FineTune => Self::FineTune,
            UploadPurpose::Vision => Self::Vision,
            UploadPurpose::UserData => Self::UserData,
            UploadPurpose::Evals => Self::Evals,
        }
    }
}

/// Deprecated file processing status.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub(crate) enum FileStatus {
    /// The file was uploaded.
    Uploaded,
    /// The file was processed.
    Processed,
    /// Processing failed.
    Error,
}

/// The constant `"file"` object discriminator.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum FileObject {
    /// `"file"`.
    File,
}

/// The constant `"list"` object discriminator.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ListObject {
    /// `"list"`.
    List,
}

/// Anchor an expiration is measured from.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ExpirationAnchor {
    /// Measured from the file's creation time.
    CreatedAt,
}

/// Sort order for file listings.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, ToSchema)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ListOrder {
    /// Oldest first.
    Asc,
    /// Newest first.
    #[default]
    Desc,
}

// -----------------------------------------------------------------------------
// Responses
// -----------------------------------------------------------------------------

/// A file stored by Praxis.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, ToSchema)]
#[schema(as = OpenAIFile)]
pub(crate) struct OpenAiFile {
    /// File identifier.
    id: String,
    /// Object type, always `file`.
    #[schema(schema_with = file_object_schema)]
    object: FileObject,
    /// Size in bytes.
    #[schema(schema_with = file_bytes_schema)]
    bytes: u64,
    /// Creation timestamp in seconds since the Unix epoch.
    #[schema(format = "unixtime")]
    created_at: i64,
    /// Expiration timestamp in seconds since the Unix epoch.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schema(format = "unixtime", nullable = false)]
    expires_at: Option<i64>,
    /// Original file name.
    filename: String,
    /// Declared purpose.
    purpose: FilePurpose,
    /// Deprecated processing status; still required by the pinned contract.
    #[schema(schema_with = file_status_schema)]
    status: FileStatus,
    /// Deprecated status detail.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schema(deprecated, nullable = false)]
    status_details: Option<String>,
}

impl OpenAiFile {
    /// Describe a finalized upload: `status` is `processed`.
    #[must_use]
    pub(crate) fn new(
        id: impl Into<String>,
        bytes: u64,
        created_at: i64,
        filename: impl Into<String>,
        purpose: FilePurpose,
    ) -> Self {
        Self {
            id: id.into(),
            object: FileObject::File,
            bytes,
            created_at,
            expires_at: None,
            filename: filename.into(),
            purpose,
            status: FileStatus::Processed,
            status_details: None,
        }
    }

    /// Set the expiration timestamp.
    #[must_use]
    pub(crate) const fn with_expires_at(mut self, expires_at: i64) -> Self {
        self.expires_at = Some(expires_at);
        self
    }

    /// Set the deprecated status detail.
    #[must_use]
    #[cfg_attr(not(test), expect(dead_code, reason = "no handler sets it yet"))]
    pub(crate) fn with_status_details(mut self, details: impl Into<String>) -> Self {
        self.status_details = Some(details.into());
        self
    }

    /// File identifier.
    #[must_use]
    pub(crate) fn id(&self) -> &str {
        &self.id
    }
}

/// One page of files.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, ToSchema)]
pub(crate) struct ListFilesResponse {
    /// Always `list`.
    #[schema(value_type = String)]
    object: ListObject,
    /// Files in this page.
    data: Vec<OpenAiFile>,
    /// First file ID, or the empty string when the page is empty.
    first_id: String,
    /// Last file ID, or the empty string when the page is empty.
    last_id: String,
    /// Whether more files follow this page.
    has_more: bool,
}

impl ListFilesResponse {
    /// Build a page, deriving `first_id` and `last_id` from `data`.
    ///
    /// The pinned schema requires both as non-null strings, so an empty page
    /// reports `""` for each.
    #[must_use]
    pub(crate) fn new(data: Vec<OpenAiFile>, has_more: bool) -> Self {
        // Cloned once per page boundary: the IDs outlive the move of `data`.
        let first_id = data.first().map(|file| file.id.clone()).unwrap_or_default();
        let last_id = data.last().map(|file| file.id.clone()).unwrap_or_default();
        Self {
            object: ListObject::List,
            data,
            first_id,
            last_id,
            has_more,
        }
    }
}

/// Result of deleting a file.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, ToSchema)]
pub(crate) struct DeleteFileResponse {
    /// Deleted file identifier.
    id: String,
    /// Object type, always `file`.
    #[schema(schema_with = file_object_schema)]
    object: FileObject,
    /// Whether the file was deleted.
    deleted: bool,
}

impl DeleteFileResponse {
    /// Report a successful deletion of `id`.
    #[must_use]
    pub(crate) fn deleted(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            object: FileObject::File,
            deleted: true,
        }
    }
}

/// Raw file bytes returned by `downloadFile` as `application/octet-stream`.
#[derive(ToSchema)]
#[schema(value_type = String, format = Binary)]
pub(crate) struct FileContent(Vec<u8>);

// -----------------------------------------------------------------------------
// Requests
// -----------------------------------------------------------------------------

/// Expiration policy for an uploaded file.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize, ToSchema)]
pub(crate) struct FileExpirationAfter {
    /// Anchor timestamp, always `created_at`.
    anchor: ExpirationAnchor,
    /// Seconds after the anchor; 3600 to 2592000.
    #[schema(value_type = i64, format = Int64, minimum = 3600, maximum = 2_592_000)]
    seconds: u32,
}

impl FileExpirationAfter {
    /// Validate bounds on an already-typed value.
    ///
    /// # Errors
    ///
    /// Returns [`InvalidField`] when `seconds` is outside 3600..=2592000.
    pub(crate) fn validate(self) -> Result<Self, InvalidField> {
        if (MIN_EXPIRATION_SECONDS..=MAX_EXPIRATION_SECONDS).contains(&self.seconds) {
            Ok(self)
        } else {
            Err(InvalidField::new(
                "expires_after[seconds]",
                format!("must be between {MIN_EXPIRATION_SECONDS} and {MAX_EXPIRATION_SECONDS}"),
            ))
        }
    }

    /// Build from the multipart form fields `expires_after[anchor]` and
    /// `expires_after[seconds]`, which SDKs send instead of JSON.
    ///
    /// # Errors
    ///
    /// Returns [`InvalidField`] for an unknown anchor, a non-integer, or an
    /// out-of-range `seconds`.
    pub(crate) fn from_form_fields(anchor: &str, seconds: &str) -> Result<Self, InvalidField> {
        if anchor != "created_at" {
            return Err(InvalidField::new(
                "expires_after[anchor]",
                format!("unsupported anchor `{anchor}`"),
            ));
        }
        let seconds = seconds
            .parse::<u32>()
            .map_err(|err| InvalidField::new("expires_after[seconds]", format!("must be an integer ({err})")))?;
        Self {
            anchor: ExpirationAnchor::CreatedAt,
            seconds,
        }
        .validate()
    }

    /// Seconds after the anchor.
    #[must_use]
    pub(crate) const fn seconds(self) -> u32 {
        self.seconds
    }
}

/// Multipart body of `createFile`. Schema only: the form is parsed
/// incrementally rather than deserialized.
#[derive(ToSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct CreateFileRequest {
    /// The file to upload.
    #[schema(value_type = String, format = Binary)]
    file: Vec<u8>,
    /// Intended purpose.
    purpose: UploadPurpose,
    /// Optional expiration policy.
    #[schema(nullable = false)]
    expires_after: Option<FileExpirationAfter>,
}

/// Query parameters of `listFiles`.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq)]
pub(crate) struct ListFilesQuery {
    /// Only return files with this purpose (a plain string upstream).
    pub(crate) purpose: Option<String>,
    /// Page size, 1..=10000; defaults to 10000.
    #[serde(default, deserialize_with = "deserialize_limit")]
    pub(crate) limit: Option<u32>,
    /// Sort order by creation time.
    #[serde(default)]
    pub(crate) order: ListOrder,
    /// Cursor: the file ID to list after.
    pub(crate) after: Option<String>,
}

impl ListFilesQuery {
    /// Effective page size.
    #[must_use]
    pub(crate) fn effective_limit(&self) -> u32 {
        self.limit.unwrap_or(MAX_LIST_LIMIT)
    }
}

/// Reject `limit` values outside the pinned range.
fn deserialize_limit<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Option<u32>, D::Error> {
    let limit = Option::<u32>::deserialize(deserializer)?;
    if let Some(value) = limit
        && !(MIN_LIST_LIMIT..=MAX_LIST_LIMIT).contains(&value)
    {
        return Err(serde::de::Error::custom(format!(
            "limit must be between {MIN_LIST_LIMIT} and {MAX_LIST_LIMIT}"
        )));
    }
    Ok(limit)
}

/// Emit the `listFiles` `limit` query schema: integer, default 10000.
///
/// The 1..=10000 range is enforced at runtime and stated in the parameter
/// description; the pinned schema carries no `minimum`/`maximum`.
pub(crate) fn list_files_limit_schema() -> RefOr<Schema> {
    RefOr::T(Schema::Object(
        ObjectBuilder::new()
            .schema_type(Type::Integer)
            .default(Some(Value::from(MAX_LIST_LIMIT)))
            .build(),
    ))
}

/// Emit the `listFiles` `order` query schema: `asc` or `desc`, default `desc`.
pub(crate) fn list_files_order_schema() -> RefOr<Schema> {
    RefOr::T(Schema::Object(
        ObjectBuilder::new()
            .schema_type(Type::String)
            .enum_values(Some(["asc", "desc"]))
            .default(Some(Value::from("desc")))
            .build(),
    ))
}

/// Inline deprecated `status` schema; a `$ref` would drop `deprecated`.
fn file_status_schema() -> Object {
    ObjectBuilder::new()
        .schema_type(Type::String)
        .enum_values(Some(["uploaded", "processed", "error"]))
        .deprecated(Some(Deprecated::True))
        .build()
}

/// Plain `integer` schema for `bytes`.
///
/// The derive would emit `format: int64, minimum: 0` for a `u64`, which the
/// pinned schema does not declare.
fn file_bytes_schema() -> Object {
    ObjectBuilder::new()
        .schema_type(Type::Integer)
        .description(Some("The size of the file, in bytes."))
        .build()
}

/// Fixed `"file"` discriminator schema.
fn file_object_schema() -> Object {
    ObjectBuilder::new()
        .schema_type(Type::String)
        .enum_values(Some(["file"]))
        .build()
}

#[cfg(test)]
#[expect(clippy::unwrap_used, clippy::indexing_slicing, reason = "tests")]
mod tests {
    use serde_json::json;

    use super::*;

    fn file(id: &str) -> OpenAiFile {
        OpenAiFile::new(id, 3, 1_700_000_000, "a.txt", FilePurpose::UserData)
    }

    #[test]
    fn file_full_golden() {
        let mut full = file("file_1").with_expires_at(1_700_003_600);
        full.status_details = Some("detail".to_owned());
        assert_eq!(
            serde_json::to_value(full).unwrap(),
            json!({
                "id": "file_1", "object": "file", "bytes": 3, "created_at": 1_700_000_000,
                "expires_at": 1_700_003_600, "filename": "a.txt", "purpose": "user_data",
                "status": "processed", "status_details": "detail"
            })
        );
    }

    #[test]
    fn file_minimal_omits_optionals_but_keeps_status() {
        let value = serde_json::to_value(file("file_1")).unwrap();
        let object = value.as_object().unwrap();
        assert!(!object.contains_key("expires_at"));
        assert!(!object.contains_key("status_details"));
        assert_eq!(object["status"], "processed");
        assert_eq!(object["object"], "file");
    }

    #[test]
    fn file_serializes_evals_purpose() {
        let evals = OpenAiFile::new("f", 0, 0, "e.jsonl", UploadPurpose::Evals.into());
        assert_eq!(serde_json::to_value(evals).unwrap()["purpose"], "evals");
    }

    #[test]
    fn list_populated_derives_first_and_last_ids() {
        let list = ListFilesResponse::new(vec![file("a"), file("b"), file("c")], true);
        let value = serde_json::to_value(list).unwrap();
        assert_eq!(value["object"], "list");
        assert_eq!(value["first_id"], "a");
        assert_eq!(value["last_id"], "c");
        assert_eq!(value["has_more"], true);
        assert_eq!(value["data"].as_array().unwrap().len(), 3);
    }

    #[test]
    fn list_empty_keeps_id_keys_as_empty_strings() {
        let value = serde_json::to_value(ListFilesResponse::new(Vec::new(), false)).unwrap();
        assert_eq!(
            value,
            json!({"object": "list", "data": [], "first_id": "", "last_id": "", "has_more": false})
        );
    }

    #[test]
    fn delete_golden() {
        assert_eq!(
            serde_json::to_value(DeleteFileResponse::deleted("file_1")).unwrap(),
            json!({"id": "file_1", "object": "file", "deleted": true})
        );
    }

    #[test]
    fn upload_purpose_accepts_every_value_and_rejects_others() {
        for value in ["assistants", "batch", "fine-tune", "vision", "user_data", "evals"] {
            let parsed = value.parse::<UploadPurpose>().unwrap();
            assert_eq!(parsed.as_str(), value);
            assert_eq!(serde_json::from_value::<UploadPurpose>(json!(value)).unwrap(), parsed);
        }
        for value in [
            "",
            "unknown",
            "Assistants",
            "USER_DATA",
            "assistants_output",
            "batch_output",
        ] {
            assert!(value.parse::<UploadPurpose>().is_err(), "{value} must be rejected");
            assert!(serde_json::from_value::<UploadPurpose>(json!(value)).is_err());
        }
    }

    #[test]
    fn expiration_bounds() {
        for (seconds, ok) in [(3_599, false), (3_600, true), (2_592_000, true), (2_592_001, false)] {
            let result = FileExpirationAfter::from_form_fields("created_at", &seconds.to_string());
            assert_eq!(result.is_ok(), ok, "{seconds}");
        }
    }

    #[test]
    fn expiration_form_field_errors() {
        let anchor = FileExpirationAfter::from_form_fields("updated_at", "3600").unwrap_err();
        assert_eq!(anchor.field, "expires_after[anchor]");
        for bad in ["abc", "", "-1", "3600.5"] {
            let err = FileExpirationAfter::from_form_fields("created_at", bad).unwrap_err();
            assert_eq!(err.field, "expires_after[seconds]", "{bad}");
        }
        let ok = FileExpirationAfter::from_form_fields("created_at", "7200").unwrap();
        assert_eq!(ok.seconds(), 7_200);
        assert_eq!(
            serde_json::to_value(ok).unwrap(),
            json!({"anchor": "created_at", "seconds": 7200})
        );
    }

    #[test]
    fn list_query_defaults() {
        let query: ListFilesQuery = serde_json::from_value(json!({})).unwrap();
        assert_eq!(query.effective_limit(), 10_000);
        assert_eq!(query.order, ListOrder::Desc);
        assert!(query.purpose.is_none() && query.after.is_none());
    }

    #[test]
    fn list_query_limit_and_order_validation() {
        for limit in [0, 10_001] {
            assert!(serde_json::from_value::<ListFilesQuery>(json!({"limit": limit})).is_err());
        }
        for limit in [1, 10_000] {
            assert!(serde_json::from_value::<ListFilesQuery>(json!({"limit": limit})).is_ok());
        }
        assert!(serde_json::from_value::<ListFilesQuery>(json!({"order": "sideways"})).is_err());
        let asc: ListFilesQuery = serde_json::from_value(json!({"order": "asc", "purpose": "batch"})).unwrap();
        assert_eq!(asc.order, ListOrder::Asc);
        assert_eq!(asc.purpose.as_deref(), Some("batch"));
    }
}
