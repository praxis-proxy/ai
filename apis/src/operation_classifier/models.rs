// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Shared HTTP routes for the OpenAI and Anthropic Models APIs.

use http::HeaderMap;

use super::{AiOperationMatch, classified};
use crate::operation::{
    ApplicationProtocol, HandlingMode, HttpMethod, OperationSpec, RequestBody, Transport, match_operation,
};

/// OpenAI operation IDs from the pinned OpenAPI document.
const OPENAI_MODELS: [OperationSpec; 2] =
    model_operations("openai_models", "listModels", "retrieveModel", "/v1/models/{model}");

/// Anthropic operation IDs from the pinned OpenAPI document.
const ANTHROPIC_MODELS: [OperationSpec; 2] =
    model_operations("anthropic_models", "models_list", "models_get", "/v1/models/{model_id}");

/// Share method and body metadata while keeping each API’s path parameter name.
const fn model_operations(
    protocol: &'static str,
    list_id: &'static str,
    retrieve_id: &'static str,
    retrieve_path: &'static str,
) -> [OperationSpec; 2] {
    let list = OperationSpec {
        application_protocol: ApplicationProtocol::new(protocol),
        operation_id: list_id,
        method: HttpMethod::Get,
        transport: Transport::Http,
        runtime_path: "/v1/models",
        mode: HandlingMode::Passthrough,
        request_body: RequestBody::None,
    };
    [
        list,
        OperationSpec {
            operation_id: retrieve_id,
            runtime_path: retrieve_path,
            ..list
        },
    ]
}

/// Match an HTTP Models operation using the client protocol signal.
pub(super) fn classify(method: &str, path: &str, headers: &HeaderMap) -> Option<AiOperationMatch> {
    let operations = if headers.contains_key("anthropic-version") {
        &ANTHROPIC_MODELS
    } else {
        &OPENAI_MODELS
    };
    let matched = match_operation(operations, method, path, Transport::Http)?;
    // Preserve fallback routing for native Gemini credentials on these shared paths.
    let google_oauth = headers.contains_key(http::header::AUTHORIZATION)
        && (headers.contains_key("x-goog-user-project") || headers.contains_key("x-goog-api-client"));
    if google_oauth
        || headers.contains_key("x-goog-api-key")
        || path
            .split_once('?')
            .is_some_and(|(_, query)| has_google_query_credentials(query))
    {
        return None;
    }
    classified(matched.spec, matched.params, path)
}

/// Inspect only query parameter names so credential values stay untouched.
fn has_google_query_credentials(query: &str) -> bool {
    query.split('&').any(|parameter| {
        let name = parameter.split_once('=').map_or(parameter, |(name, _)| name);
        url::form_urlencoded::parse(name.as_bytes())
            .any(|(name, _)| matches!(name.as_ref(), "key" | "access_token" | "oauth_token"))
    })
}
