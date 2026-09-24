// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Vertex AI request transformation.
//!
//! Anthropic-dialect requests (`POST /v1/messages` with `model` and an
//! optional `stream` flag) are rewritten into the Vertex `rawPredict`
//! wire shape:
//!
//! - the model moves from the body into the URL path (`…/publishers/anthropic/models/{model}:rawPredict`, or
//!   `:streamRawPredict` when `stream` is `true`) — Vertex rejects an unknown body `model` field with `model: Extra
//!   inputs are not permitted`, so it must be removed;
//! - `anthropic_version` is injected with the fixed Vertex value;
//! - `count_tokens` keeps its `model` in the body but at a distinct URL (`…/models/count-tokens:rawPredict`).
//!
//! The `stream` flag is read *before* the body is replaced, since it
//! selects the URL verb.

use serde_json::Value;

use super::config::{VertexConfig, is_safe_model_char};

/// Body value of `anthropic_version` accepted by Vertex `rawPredict`.
pub(crate) const VERTEX_ANTHROPIC_VERSION: &str = "vertex-2023-10-16";

/// Anthropic Messages endpoint paths handled by this filter.
pub(crate) const MESSAGES_PATH: &str = "/v1/messages";
/// Anthropic token-counting endpoint path handled by this filter.
pub(crate) const COUNT_TOKENS_PATH: &str = "/v1/messages/count_tokens";

/// The Anthropic operation a request targets.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Operation {
    /// `POST /v1/messages`
    Messages,
    /// `POST /v1/messages/count_tokens`
    CountTokens,
}

/// Classify a request path into a Vertex-supported Anthropic operation.
/// Anything else is passed through untouched.
pub(crate) fn classify(path: &str) -> Option<Operation> {
    let path_only = path.split_once('?').map_or(path, |(p, _)| p);
    match path_only {
        COUNT_TOKENS_PATH => Some(Operation::CountTokens),
        MESSAGES_PATH => Some(Operation::Messages),
        _ => None,
    }
}

/// Why a request could not be transformed; each maps to an Anthropic
/// `invalid_request_error` rejection.
#[derive(Debug, Eq, PartialEq)]
pub(crate) enum RequestError {
    /// Body was not a valid JSON object; carries the serde detail.
    InvalidJson(String),
    /// Body had no string `model` field.
    MissingModel,
    /// Model id carried characters unsafe for the URL path.
    UnsafeModel(String),
}

impl std::fmt::Display for RequestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidJson(detail) => write!(f, "request body must be a JSON object: {detail}"),
            Self::MissingModel => f.write_str("request body must carry a string 'model' field"),
            Self::UnsafeModel(model) => write!(
                f,
                "model '{model}' contains characters not allowed in a Vertex model id \
                 (letters, digits, '.', '-', '_', '@')"
            ),
        }
    }
}

impl std::error::Error for RequestError {}

/// The rewritten request: body bytes, upstream path, and the original
/// user-facing model id (needed to restore the model in responses).
#[derive(Debug)]
pub(crate) struct TransformedRequest {
    /// Transformed request body.
    pub body: Vec<u8>,
    /// Vertex `rawPredict` path to send upstream.
    pub path: String,
    /// User-facing model id as the client sent it (prefix included).
    pub user_model: String,
}

/// Strip the user-facing prefix to obtain the Vertex publisher model id.
/// Models sent without the prefix (e.g. plain `claude-sonnet-4-5`) pass
/// through unchanged, so a gateway can expose the same id on both the
/// direct-Anthropic and Vertex backends.
fn publisher_model<'a>(user_model: &'a str, prefix: &str) -> &'a str {
    user_model.strip_prefix(prefix).unwrap_or(user_model)
}

/// Validate a publisher model id against the URL-path character
/// allowlist (non-empty, no path/query/verb injection).
fn validate_publisher_model(model: &str) -> Result<(), RequestError> {
    if model.is_empty() || !model.chars().all(is_safe_model_char) {
        return Err(RequestError::UnsafeModel(model.to_owned()));
    }
    Ok(())
}

/// Transform an Anthropic request body + path into Vertex form.
///
/// # Errors
///
/// Returns [`RequestError`] when the body is not a JSON object, carries
/// no string `model`, or the model id is unsafe for the URL path.
pub(crate) fn transform_request(
    body: &[u8],
    operation: Operation,
    cfg: &VertexConfig,
) -> Result<Option<TransformedRequest>, RequestError> {
    let mut value: Value =
        serde_json::from_slice(body).map_err(|error| RequestError::InvalidJson(error.to_string()))?;
    let obj = value
        .as_object_mut()
        .ok_or_else(|| RequestError::InvalidJson("expected a JSON object".to_owned()))?;

    let user_model = obj
        .get("model")
        .and_then(Value::as_str)
        .ok_or(RequestError::MissingModel)?
        .to_owned();
    if !user_model.starts_with(&cfg.model_prefix) {
        return Ok(None);
    }
    let publisher = publisher_model(&user_model, &cfg.model_prefix);
    validate_publisher_model(publisher)?;

    let path = match operation {
        Operation::Messages => transform_messages(obj, publisher, cfg),
        Operation::CountTokens => {
            // count_tokens takes the model in the body, unlike
            // :rawPredict which takes it in the URL.
            obj.insert("model".to_owned(), Value::String(publisher.to_owned()));
            format!(
                "/v1/projects/{}/locations/{}/publishers/anthropic/models/count-tokens:rawPredict",
                cfg.project, cfg.location
            )
        },
    };

    let body = serde_json::to_vec(&value)
        .map_err(|error| RequestError::InvalidJson(format!("re-serializing the request failed: {error}")))?;
    Ok(Some(TransformedRequest { body, path, user_model }))
}

/// Rewrite the Messages body in place and build its `rawPredict` path.
fn transform_messages(obj: &mut serde_json::Map<String, Value>, publisher: &str, cfg: &VertexConfig) -> String {
    // The stream verb must be decided from the body's `stream` flag
    // before anything else mutates it; after the rewrite nothing else
    // carries it into the URL.
    let verb = if obj.get("stream").and_then(Value::as_bool).unwrap_or(false) {
        ":streamRawPredict"
    } else {
        ":rawPredict"
    };
    // Vertex rejects an unknown body `model` field with
    // `model: Extra inputs are not permitted`.
    obj.remove("model");
    obj.insert(
        "anthropic_version".to_owned(),
        Value::String(VERTEX_ANTHROPIC_VERSION.to_owned()),
    );
    format!(
        "/v1/projects/{}/locations/{}/publishers/anthropic/models/{}{verb}",
        cfg.project,
        cfg.location,
        model_with_pin(publisher, cfg)
    )
}

/// Publisher model with the configured snapshot pin appended, if any.
fn model_with_pin(publisher: &str, cfg: &VertexConfig) -> String {
    match &cfg.model_pin {
        Some(pin) => format!("{publisher}{pin}"),
        None => publisher.to_owned(),
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, clippy::indexing_slicing, reason = "tests")]
mod tests {
    use serde_json::json;

    use super::*;

    fn cfg() -> VertexConfig {
        let yaml: serde_yaml::Value = serde_yaml::from_str("project: demo-project\nmodel_pin: '@20250929'").unwrap();
        let cfg: VertexConfig = serde_yaml::from_value(yaml).unwrap();
        crate::vertex::config::build_config(cfg).unwrap()
    }

    #[test]
    fn classify_paths() {
        assert_eq!(classify("/v1/messages"), Some(Operation::Messages));
        assert_eq!(classify("/v1/messages?alt=json"), Some(Operation::Messages));
        assert_eq!(classify("/v1/messages/count_tokens"), Some(Operation::CountTokens));
        assert_eq!(classify("/v1/messages/batches"), None);
        assert_eq!(classify("/v1/models"), None);
    }

    #[test]
    fn messages_move_model_to_url_and_inject_version() {
        let body = json!({"model": "vertex/claude-sonnet-4-5", "max_tokens": 8, "messages": []}).to_string();
        let out = transform_request(body.as_bytes(), Operation::Messages, &cfg())
            .unwrap()
            .unwrap();

        assert_eq!(
            out.path,
            "/v1/projects/demo-project/locations/global/publishers/anthropic/models/claude-sonnet-4-5@20250929:rawPredict"
        );
        assert_eq!(out.user_model, "vertex/claude-sonnet-4-5");
        let parsed: Value = serde_json::from_slice(&out.body).unwrap();
        assert!(
            parsed.get("model").is_none(),
            "Vertex rejects a body model field; it must be stripped"
        );
        assert_eq!(parsed["anthropic_version"], VERTEX_ANTHROPIC_VERSION);
        assert_eq!(parsed["max_tokens"], 8, "other fields must pass through");
    }

    #[test]
    fn stream_flag_selects_stream_verb() {
        let body = json!({"model": "vertex/claude-sonnet-4-5", "stream": true, "messages": []}).to_string();
        let out = transform_request(body.as_bytes(), Operation::Messages, &cfg()).unwrap();
        let out = out.unwrap();
        assert!(out.path.ends_with(":streamRawPredict"), "got {}", out.path);
        assert!(
            out.path.contains("models/claude-sonnet-4-5@20250929"),
            "got {}",
            out.path
        );
    }

    #[test]
    fn non_vertex_models_are_not_transformed() {
        let body = json!({"model": "claude-sonnet-5", "messages": []}).to_string();
        assert!(
            transform_request(body.as_bytes(), Operation::Messages, &cfg())
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn count_tokens_keeps_model_in_body() {
        let body = json!({"model": "vertex/claude-sonnet-4-5", "messages": []}).to_string();
        let out = transform_request(body.as_bytes(), Operation::CountTokens, &cfg())
            .unwrap()
            .unwrap();
        assert_eq!(
            out.path,
            "/v1/projects/demo-project/locations/global/publishers/anthropic/models/count-tokens:rawPredict"
        );
        let parsed: Value = serde_json::from_slice(&out.body).unwrap();
        assert_eq!(
            parsed["model"], "claude-sonnet-4-5",
            "count_tokens takes the publisher model in the body"
        );
    }

    #[test]
    fn rejects_model_path_injection() {
        for model in [
            "vertex/../../secrets",
            "vertex/model:rawPredict",
            "vertex/model?x=1",
            "vertex/model#f",
            "vertex/model name",
            "vertex/",
        ] {
            let body = json!({"model": model, "messages": []}).to_string();
            let out = transform_request(body.as_bytes(), Operation::Messages, &cfg());
            assert!(
                matches!(out, Err(RequestError::UnsafeModel(_))),
                "model {model:?} must be rejected, got {out:?}"
            );
        }
    }

    #[test]
    fn rejects_missing_model_and_non_objects() {
        let missing = transform_request(br#"{"messages":[]}"#, Operation::Messages, &cfg());
        assert_eq!(missing.err(), Some(RequestError::MissingModel));
        let bad = transform_request(b"not json", Operation::Messages, &cfg());
        assert!(matches!(bad.err(), Some(RequestError::InvalidJson(_))));
        let array = transform_request(b"[1,2]", Operation::Messages, &cfg());
        assert!(matches!(array.err(), Some(RequestError::InvalidJson(_))));
    }
}
