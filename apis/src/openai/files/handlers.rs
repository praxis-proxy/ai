// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Query parsing and error rendering for the local Files handlers.
//!
//! The DTO-to-wire conversions live in `mapping.rs`.

use http::{HeaderMap, HeaderValue, header::CONTENT_TYPE};
use praxis_filter::{FilterError, TerminalResponse};
use serde::{Deserialize as _, Serialize};
use serde_json::{Map, Value};

use super::{FileServiceError, contracts::ListFilesQuery};
use crate::openai::{error::ErrorResponse, include::decode_query_component_strict};

/// A rejected query parameter.
#[derive(Debug, Eq, PartialEq)]
pub(super) struct InvalidQuery {
    /// Offending parameter, or `None` when the query string itself is malformed.
    pub(super) param: Option<String>,
    /// Client-facing explanation.
    pub(super) message: String,
}

/// Parse the `listFiles` query string. Unknown parameters are ignored and
/// the last occurrence of a repeated parameter wins. `limit` must be an
/// integer in 1..=10000.
pub(super) fn parse_list_query(query: Option<&str>) -> Result<ListFilesQuery, InvalidQuery> {
    let mut fields = Map::new();
    for pair in query.unwrap_or_default().split('&').filter(|pair| !pair.is_empty()) {
        let (raw_key, raw_value) = pair.split_once('=').unwrap_or((pair, ""));
        let key = decode_query_component_strict(raw_key).map_err(malformed)?;
        if !matches!(key.as_ref(), "purpose" | "limit" | "order" | "after") {
            continue;
        }
        let value = decode_query_component_strict(raw_value).map_err(malformed)?;
        let value = if key == "limit" {
            let limit = value.parse::<u32>().map_err(|_e| InvalidQuery {
                param: Some("limit".to_owned()),
                message: format!("Invalid value for 'limit': '{value}' is not a valid integer."),
            })?;
            Value::from(limit)
        } else {
            Value::String(value.into_owned())
        };
        fields.insert(key.into_owned(), value);
    }
    ListFilesQuery::deserialize(Value::Object(fields.clone())).map_err(|error| InvalidQuery {
        // Serde errors do not name the field; re-check each one alone.
        param: fields
            .iter()
            .find(|(key, value)| single_field_fails(key, value))
            .map(|(key, _value)| key.clone()),
        message: error.to_string(),
    })
}

/// Build the error for an undecodable query string.
fn malformed(message: String) -> InvalidQuery {
    InvalidQuery { param: None, message }
}

/// Whether the query is rejected with only this field present.
fn single_field_fails(key: &str, value: &Value) -> bool {
    let single = Map::from_iter([(key.to_owned(), value.clone())]);
    ListFilesQuery::deserialize(Value::Object(single)).is_err()
}

/// Render a rejected query as a 400.
pub(super) fn invalid_query_response(invalid: &InvalidQuery) -> Result<TerminalResponse, FilterError> {
    json_response(
        400,
        &ErrorResponse::new(
            &invalid.message,
            "invalid_request_error",
            None,
            invalid.param.as_deref(),
        ),
    )
}

/// Render a service failure. `file_id` names the requested file for retrieve.
///
/// Catalog and store details never reach the client: they can carry
/// connection strings or SQL text.
pub(super) fn error_response(error: &FileServiceError, file_id: Option<&str>) -> Result<TerminalResponse, FilterError> {
    match error {
        FileServiceError::NotFound => {
            let message = file_id.map_or_else(
                || "No such File object.".to_owned(),
                |id| format!("No such File object: {id}"),
            );
            json_response(
                404,
                &ErrorResponse::new(&message, "invalid_request_error", None, Some("id")),
            )
        },
        FileServiceError::InvalidInput(message) => {
            json_response(400, &ErrorResponse::new(message, "invalid_request_error", None, None))
        },
        // Upload, quota and stream failures cannot occur for list/retrieve;
        // fail closed rather than invent a client status for them.
        FileServiceError::Catalog(_)
        | FileServiceError::Store(_)
        | FileServiceError::UploadTooLarge
        | FileServiceError::QuotaExceeded
        | FileServiceError::Stream(_) => json_response(
            500,
            &ErrorResponse::new("Internal server error.", "server_error", None, None),
        ),
    }
}

/// Serialize `body` as a buffered JSON response.
fn json_response<T: Serialize>(status: u16, body: &T) -> Result<TerminalResponse, FilterError> {
    let bytes = serde_json::to_vec(body)
        .map_err(|error| FilterError::from(format!("openai_files: serialize failed: {error}")))?;
    let mut headers = HeaderMap::new();
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    Ok(TerminalResponse::new(status).with_headers(headers).with_body(bytes))
}

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::unwrap_used, clippy::indexing_slicing, reason = "tests")]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::openai::files::{FileList, FileListOrder};

    fn body(response: &TerminalResponse) -> Value {
        serde_json::from_slice(response.body.as_ref().unwrap()).unwrap()
    }

    #[test]
    fn query_defaults_and_values() {
        let options = FileList::from(parse_list_query(None).unwrap());
        assert_eq!(
            options,
            FileList {
                purpose: None,
                limit: 10_000,
                order: FileListOrder::Desc,
                after: None
            }
        );
        let options =
            FileList::from(parse_list_query(Some("limit=5&order=asc&purpose=batch&after=file%5F1&unknown=x")).unwrap());
        assert_eq!(
            options,
            FileList {
                purpose: Some("batch".to_owned()),
                limit: 5,
                order: FileListOrder::Asc,
                after: Some("file_1".to_owned())
            }
        );
    }

    #[test]
    fn invalid_query_names_the_field() {
        for (query, param) in [
            ("limit=0", Some("limit")),
            ("limit=10001", Some("limit")),
            ("limit=-1", Some("limit")),
            ("limit=abc", Some("limit")),
            ("order=sideways", Some("order")),
            ("purpose=batch&order=up&limit=5", Some("order")),
            ("after=%FF", None),
        ] {
            let invalid = parse_list_query(Some(query)).unwrap_err();
            assert_eq!(invalid.param.as_deref(), param, "{query}");
            let response = invalid_query_response(&invalid).unwrap();
            assert_eq!(response.status, 400);
            let value = body(&response);
            assert_eq!(value["error"]["type"], "invalid_request_error");
            assert_eq!(value["error"]["param"], json!(param));
        }
        assert!(parse_list_query(Some("limit=1")).is_ok());
        assert!(parse_list_query(Some("limit=10000")).is_ok());
    }

    #[test]
    fn error_statuses_and_bodies() {
        let not_found = error_response(&FileServiceError::NotFound, Some("file_x")).unwrap();
        assert_eq!(not_found.status, 404);
        assert_eq!(
            body(&not_found),
            json!({"error": {"code": null, "message": "No such File object: file_x",
                             "param": "id", "type": "invalid_request_error"}})
        );
        assert_eq!(not_found.headers.get(CONTENT_TYPE).unwrap(), "application/json");

        let invalid = error_response(
            &FileServiceError::InvalidInput("invalid `after` cursor".to_owned()),
            None,
        )
        .unwrap();
        assert_eq!(invalid.status, 400);
        assert_eq!(body(&invalid)["error"]["type"], "invalid_request_error");

        let secret = "postgres-secret";
        let internal = [
            FileServiceError::Catalog(praxis_ai_store::CatalogError::Backend(secret.into())),
            FileServiceError::Store(super::super::FileStoreError::Backend(secret.into())),
            FileServiceError::QuotaExceeded,
        ];
        for error in &internal {
            let response = error_response(error, None).unwrap();
            assert_eq!(response.status, 500);
            let value = body(&response);
            assert_eq!(value["error"]["type"], "server_error");
            assert_eq!(value["error"]["message"], "Internal server error.");
            assert!(!String::from_utf8_lossy(response.body.as_ref().unwrap()).contains("hunter2"));
        }
    }
}
