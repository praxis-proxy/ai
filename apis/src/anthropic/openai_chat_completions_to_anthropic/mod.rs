// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Provider-selected OpenAI Chat Completions to Anthropic Messages conversion.
//!
//! The filter consumes only the trusted profile written by
//! `intelligent_route` after selecting a candidate. It never infers provider
//! format from the client model or a client header. The selected profile is
//! copied into request-scoped metadata before body processing and is the only
//! source used again by response conversion.

mod config;
mod request;
mod response;

use async_trait::async_trait;
use bytes::Bytes;
use http::{HeaderName, HeaderValue};
use praxis_filter::{
    BodyAccess, BodyMode, FilterAction, FilterError, HttpFilter, HttpFilterContext, Rejection,
    SelectedUpstreamBodyOutcome, parse_filter_config,
};
pub(crate) use response::{map_stop_reason, total_input_tokens};

use self::config::OpenaiChatCompletionsToAnthropicConfig;

/// Registered filter identifier.
const FILTER_NAME: &str = "openai_chat_completions_to_anthropic";
/// Trusted request outcome written by `intelligent_route`.
const ROUTE_OUTCOME: &str = "intelligent_route.outcome";
/// Stable candidate identifier written by `intelligent_route`.
const ROUTE_STABLE_ID: &str = "intelligent_route.stable_id";
/// Content-addressed overlay revision used for the selection.
const ROUTE_OVERLAY_REVISION: &str = "intelligent_route.overlay_revision";
/// Selected provider API format.
const ROUTE_API_FORMAT: &str = "intelligent_route.api_format";
/// Selected provider request path.
const ROUTE_PATH: &str = "intelligent_route.path";
/// Selected provider target model.
const ROUTE_TARGET_MODEL: &str = "intelligent_route.target_model";
/// Selected `ExternalModel`'s client-visible name.
const ROUTE_NAME: &str = "intelligent_route.name";
/// Selected credential strategy.
const CREDENTIAL_STRATEGY: &str = "intelligent_route.credential.strategy";
/// Selected credential Secret name.
const CREDENTIAL_NAME: &str = "intelligent_route.credential.name";
/// Selected credential Secret namespace.
const CREDENTIAL_NAMESPACE: &str = "intelligent_route.credential.namespace";
/// Selected credential Secret key.
const CREDENTIAL_KEY: &str = "intelligent_route.credential.key";
/// Prefix for the request-bound profile fields.
pub(super) const BOUND_PREFIX: &str = "openai_chat_completions_to_anthropic.bound.";
/// Request's selected streaming flag.
pub(super) const REQUEST_STREAM: &str = "openai_chat_completions_to_anthropic.request_stream";
/// Client-visible model saved for response serialization.
pub(super) const CLIENT_MODEL: &str = "openai_chat_completions_to_anthropic.client_model";
/// OpenAI streaming usage preference saved for SSE conversion.
pub(super) const INCLUDE_USAGE: &str = "openai_chat_completions_to_anthropic_stream.include_usage";
/// Marker indicating that JSON response conversion was selected.
const RESPONSE_ARMED: &str = "openai_chat_completions_to_anthropic.response_armed";
/// Marker indicating that the selected upstream returned an error response.
const RESPONSE_ERROR: &str = "openai_chat_completions_to_anthropic.response_error";
/// Upstream status saved before the body phase loses response-header access.
const RESPONSE_STATUS: &str = "openai_chat_completions_to_anthropic.response_status";

/// Immutable selected provider profile carried from the request to its response.
#[derive(Clone, Debug, Eq, PartialEq)]
struct SelectedProfile {
    /// Provider candidate identity.
    stable_id: String,
    /// Serving overlay revision.
    revision: String,
    /// Provider request format.
    api_format: String,
    /// Provider request path.
    path: String,
    /// Provider target model.
    target_model: String,
    /// Client-visible `ExternalModel` name.
    candidate_name: String,
    /// Selected Secret injection strategy.
    credential_strategy: Option<String>,
    /// Selected Secret name.
    credential_name: Option<String>,
    /// Selected Secret namespace.
    credential_namespace: Option<String>,
    /// Selected Secret key.
    credential_key: Option<String>,
}

/// Bounded response bytes retained for one JSON response on one request.
#[derive(Default)]
struct BufferedJsonResponse {
    /// Provider JSON chunks collected until end of stream.
    bytes: Vec<u8>,
}

/// Translates OpenAI Chat Completions requests only when the selected
/// `ExternalModel` candidate uses Anthropic Messages; OpenAI candidates pass
/// through with their configured path and model.
///
/// The trusted profile is emitted by `intelligent_route`, never inferred from
/// a request header or model name. The selected Secret is injected later by
/// `credential_inject`.
///
/// For Anthropic candidates, non-streaming JSON response chunks are retained in
/// bounded request-local state until they can be translated. The separate SSE
/// filter converts event chunks incrementally, and OpenAI-compatible response
/// chunks remain unchanged.
///
/// Anthropic translation runs only when the filter's `api_translation_enabled`
/// setting is true, as derived from the tenant's API translation permission.
///
/// ```yaml
/// filter: openai_chat_completions_to_anthropic
/// api_translation_enabled: true
/// max_body_bytes: 1048576
/// ```
///
/// Only Chat Completions fields represented by Anthropic Messages are
/// accepted. Unsupported fields and modalities fail with a 400 response.
pub struct OpenaiChatCompletionsToAnthropicFilter {
    /// Filter body-processing limits.
    config: OpenaiChatCompletionsToAnthropicConfig,
}

impl OpenaiChatCompletionsToAnthropicFilter {
    /// Create this filter from its YAML configuration.
    ///
    /// # Errors
    ///
    /// Returns an error for unknown configuration keys or an invalid body limit.
    pub fn from_config(config: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
        let config: OpenaiChatCompletionsToAnthropicConfig = parse_filter_config(FILTER_NAME, config)?;
        Ok(Box::new(Self {
            config: config.validate()?,
        }))
    }

    /// Freeze the selected profile in request-scoped metadata for response use.
    fn bind_profile(ctx: &mut HttpFilterContext<'_>, profile: &SelectedProfile) -> Result<(), FilterError> {
        for (key, value) in profile_metadata(profile) {
            let bound_key = format!("{BOUND_PREFIX}{key}");
            if let Some(previous) = ctx.get_metadata(&bound_key)
                && previous != value
            {
                return Err(format!("{FILTER_NAME}: selected provider profile changed during the request").into());
            }
            ctx.set_metadata(&bound_key, value);
        }
        Ok(())
    }

    /// Check that the selection still matches the immutable request-bound profile.
    fn profile_matches_bound(ctx: &HttpFilterContext<'_>, profile: &SelectedProfile) -> bool {
        profile_metadata(profile).into_iter().all(|(key, value)| {
            let bound_key = format!("{BOUND_PREFIX}{key}");
            ctx.get_metadata(&bound_key) == Some(value)
        })
    }

    /// Build an OpenAI-compatible error response for a failed translation.
    fn openai_error_rejection(status: u16, code: &str, message: &str) -> Rejection {
        let body = serde_json::json!({"error":{"message":message,"type":code,"code":code}});
        let bytes = serde_json::to_vec(&body)
            .unwrap_or_else(|_| b"{\"error\":{\"message\":\"translation failed\",\"type\":\"server_error\"}}".to_vec());
        Rejection::status(status)
            .with_header("content-type", "application/json")
            .with_body(bytes)
            .preserving_keepalive()
    }

    /// Apply the selected provider path and Anthropic request headers.
    fn set_path_and_headers(ctx: &mut HttpFilterContext<'_>, profile: &SelectedProfile) {
        let mut path = profile.path.clone();
        if let Some(query) = ctx.request.uri.query() {
            path.push('?');
            path.push_str(query);
        }
        ctx.rewritten_path = Some(path);
        if profile.api_format == "messages" {
            ctx.request_headers_to_remove
                .push(HeaderName::from_static("authorization"));
            ctx.request_headers_to_remove.push(HeaderName::from_static("x-api-key"));
            ctx.request_headers_to_remove.push(http::header::ACCEPT_ENCODING);
            ctx.request_headers_to_set.push((
                HeaderName::from_static("anthropic-version"),
                HeaderValue::from_static("2023-06-01"),
            ));
            ctx.request_headers_to_set
                .push((http::header::CONTENT_TYPE, HeaderValue::from_static("application/json")));
        }
    }
}

#[expect(
    clippy::too_many_lines,
    reason = "The callbacks keep selected-profile validation and body limits at their enforcement boundary"
)]
#[async_trait]
impl HttpFilter for OpenaiChatCompletionsToAnthropicFilter {
    /// Return the registry name used by the `ExtProc` filter chain.
    fn name(&self) -> &'static str {
        "openai_chat_completions_to_anthropic"
    }

    /// Defer request-body mutation until `intelligent_route` has selected a provider.
    fn selected_upstream_request_body_access(&self) -> BodyAccess {
        BodyAccess::ReadWrite
    }

    /// Buffer a complete request within the configured cap before JSON translation.
    fn request_body_mode(&self) -> BodyMode {
        BodyMode::StreamBuffer {
            max_bytes: Some(self.config.max_body_bytes),
        }
    }

    /// Allow translation of complete Anthropic JSON responses.
    fn response_body_access(&self) -> BodyAccess {
        BodyAccess::ReadWrite
    }

    /// Receive response chunks while retaining JSON only in this filter's bounded request state.
    fn response_body_mode(&self) -> BodyMode {
        BodyMode::Stream
    }

    /// Validate the selected provider profile and prepare route, credential, and body handling.
    async fn on_request(&self, ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        let profile = match selected_profile(ctx) {
            Ok(profile) => profile,
            Err(error) => {
                return Ok(FilterAction::Reject(Self::openai_error_rejection(
                    503,
                    "provider_profile_unavailable",
                    &error,
                )));
            },
        };
        if profile.api_format == "messages" && !self.config.api_translation_enabled {
            return Ok(FilterAction::Reject(Self::openai_error_rejection(
                503,
                "api_translation_disabled",
                "The platform administrator has forbidden translation to this provider's API.",
            )));
        }
        if let Err(error) = Self::bind_profile(ctx, &profile) {
            return Ok(FilterAction::Reject(Self::openai_error_rejection(
                503,
                "provider_profile_changed",
                &error.to_string(),
            )));
        }
        Self::set_path_and_headers(ctx, &profile);
        Ok(FilterAction::Continue)
    }

    /// Translate the buffered request body after the selected upstream is known.
    async fn on_selected_upstream_request_body(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        body: &mut Option<Bytes>,
    ) -> Result<SelectedUpstreamBodyOutcome, FilterError> {
        let profile = match selected_profile(ctx) {
            Ok(profile) => profile,
            Err(error) => {
                return Ok(SelectedUpstreamBodyOutcome::Reject(Self::openai_error_rejection(
                    503,
                    "provider_profile_unavailable",
                    &error,
                )));
            },
        };
        if !Self::profile_matches_bound(ctx, &profile) {
            return Ok(SelectedUpstreamBodyOutcome::Reject(Self::openai_error_rejection(
                503,
                "provider_profile_changed",
                "selected provider profile changed before request translation",
            )));
        }
        let Some(bytes) = body.as_ref().filter(|bytes| !bytes.is_empty()) else {
            return Ok(SelectedUpstreamBodyOutcome::Reject(Self::openai_error_rejection(
                400,
                "invalid_request_error",
                "Chat Completions request body is empty",
            )));
        };
        if bytes.len() > self.config.max_body_bytes {
            return Ok(SelectedUpstreamBodyOutcome::Reject(Self::openai_error_rejection(
                413,
                "request_too_large",
                "Chat Completions request body exceeds a size limit set by the platform administrator.",
            )));
        }
        let client_model = match request::client_model(bytes) {
            Ok(model) => model,
            Err(error) => {
                return Ok(SelectedUpstreamBodyOutcome::Reject(Self::openai_error_rejection(
                    400,
                    "invalid_request_error",
                    &error,
                )));
            },
        };
        if client_model != profile.candidate_name {
            return Ok(SelectedUpstreamBodyOutcome::Reject(Self::openai_error_rejection(
                400,
                "invalid_request_error",
                "request model does not match the selected ExternalModel candidate",
            )));
        }
        if profile.api_format == "openai-chat" {
            match request::rewrite_openai_target_model(bytes, &profile.target_model) {
                Ok(translated) => *body = Some(Bytes::from(translated)),
                Err(error) => {
                    return Ok(SelectedUpstreamBodyOutcome::Reject(Self::openai_error_rejection(
                        400,
                        "invalid_request_error",
                        &error,
                    )));
                },
            }
            return Ok(SelectedUpstreamBodyOutcome::Continue);
        }
        let (translated, request) = match request::translate_request(bytes, &profile.target_model) {
            Ok(result) => result,
            Err(error) => {
                return Ok(SelectedUpstreamBodyOutcome::Reject(Self::openai_error_rejection(
                    400,
                    "invalid_request_error",
                    &error,
                )));
            },
        };
        ctx.set_metadata(REQUEST_STREAM, request.stream.to_string());
        ctx.set_metadata(CLIENT_MODEL, request.client_model);
        ctx.set_metadata(INCLUDE_USAGE, request.include_usage.to_string());
        *body = Some(Bytes::from(translated));
        Ok(SelectedUpstreamBodyOutcome::Continue)
    }

    /// Arm JSON response conversion only for a successful Anthropic Messages response.
    async fn on_response(&self, ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        if ctx.get_metadata(&format!("{BOUND_PREFIX}api_format")) != Some("messages") {
            return Ok(FilterAction::Continue);
        }
        let Some(response) = ctx.response_header.as_ref() else {
            return Err("openai_chat_completions_to_anthropic: response headers are missing".into());
        };
        let status = response.status.as_u16();
        let success = response.status.is_success();
        let is_event_stream = response
            .headers
            .get(http::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .is_some_and(crate::is_event_stream_content_type);
        if success && is_event_stream {
            if ctx.get_metadata(REQUEST_STREAM) == Some("true") {
                return Ok(FilterAction::Continue);
            }
            return Ok(FilterAction::Reject(Self::openai_error_rejection(
                502,
                "server_error",
                "Anthropic provider streamed a response for a non-streaming request",
            )));
        }
        if success && ctx.get_metadata(REQUEST_STREAM) == Some("true") {
            return Ok(FilterAction::Reject(Self::openai_error_rejection(
                502,
                "server_error",
                "Anthropic provider returned a non-streaming response for a streaming request",
            )));
        }
        if response.headers.contains_key(http::header::CONTENT_ENCODING) {
            return Ok(FilterAction::Reject(Self::openai_error_rejection(
                502,
                "server_error",
                "compressed Anthropic responses cannot be translated",
            )));
        }
        if let Some(response) = &mut ctx.response_header {
            response.headers.remove(http::header::CONTENT_LENGTH);
            response.headers.remove("anthropic-version");
            response
                .headers
                .insert(http::header::CONTENT_TYPE, HeaderValue::from_static("application/json"));
            ctx.response_headers_modified = true;
        }
        ctx.set_metadata(RESPONSE_ARMED, "true");
        ctx.set_metadata(RESPONSE_ERROR, (!success).to_string());
        ctx.set_metadata(RESPONSE_STATUS, status.to_string());
        Ok(FilterAction::Continue)
    }

    /// Bound and collect response chunks, then convert the complete JSON document at end of stream.
    fn on_response_body(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        body: &mut Option<Bytes>,
        end_of_stream: bool,
    ) -> Result<FilterAction, FilterError> {
        if ctx.get_metadata(RESPONSE_ARMED) != Some("true") {
            return Ok(FilterAction::Continue);
        }

        let chunk = body.as_ref().map_or(&[][..], Bytes::as_ref);
        let current_len = ctx
            .get_filter_state::<BufferedJsonResponse>()
            .map_or(0, |state| state.bytes.len());
        if current_len.saturating_add(chunk.len()) > self.config.max_body_bytes {
            ctx.remove_filter_state::<BufferedJsonResponse>();
            return Ok(FilterAction::Reject(Self::openai_error_rejection(
                502,
                "provider_response_too_large",
                "Anthropic provider response exceeds a size limit set by the platform administrator.",
            )));
        }

        if ctx.get_filter_state::<BufferedJsonResponse>().is_none() {
            ctx.insert_filter_state(BufferedJsonResponse::default());
        }
        let state = ctx
            .get_filter_state_mut::<BufferedJsonResponse>()
            .ok_or("Anthropic JSON response buffer state is unavailable")?;
        state.bytes.extend_from_slice(chunk);
        // The selected JSON response is emitted only after it has been
        // translated. SSE uses the separate incremental filter and is never
        // retained here.
        *body = Some(Bytes::new());
        if !end_of_stream {
            return Ok(FilterAction::Continue);
        }

        let buffered = ctx
            .remove_filter_state::<BufferedJsonResponse>()
            .ok_or("Anthropic JSON response buffer state disappeared at end of stream")?;
        let raw = buffered.bytes.as_slice();
        let translated = if ctx.get_metadata(RESPONSE_ERROR) == Some("true") {
            let status = ctx
                .get_metadata(RESPONSE_STATUS)
                .and_then(|value| value.parse().ok())
                .unwrap_or(502);
            response::translate_error(raw, status)
        } else {
            let client_model = ctx.get_metadata(CLIENT_MODEL).unwrap_or("unknown");
            match response::translate_response(raw, client_model) {
                Ok(bytes) => bytes,
                Err(error) => {
                    return Ok(FilterAction::Reject(Self::openai_error_rejection(
                        502,
                        "server_error",
                        &error,
                    )));
                },
            }
        };
        *body = Some(Bytes::from(translated));
        Ok(FilterAction::Continue)
    }
}

/// Resolve and validate the trusted selected candidate metadata for this request.
fn selected_profile(ctx: &HttpFilterContext<'_>) -> Result<SelectedProfile, String> {
    if !matches!(ctx.get_metadata(ROUTE_OUTCOME), Some("routed" | "reused")) {
        return Err("intelligent_route did not make a request-scoped provider selection".to_owned());
    }
    let stable_id = required_metadata(ctx, ROUTE_STABLE_ID)?;
    let revision = required_metadata(ctx, ROUTE_OVERLAY_REVISION)?;
    let api_format = required_metadata(ctx, ROUTE_API_FORMAT)?;
    let path = required_metadata(ctx, ROUTE_PATH)?;
    let target_model = required_metadata(ctx, ROUTE_TARGET_MODEL)?;
    let candidate_name = required_metadata(ctx, ROUTE_NAME)?;
    validate_profile_fields(&api_format, &revision, &path, &target_model)?;
    let credential = selected_credential(ctx);
    validate_profile_credential(&api_format, &credential)?;
    Ok(SelectedProfile {
        stable_id,
        revision,
        api_format,
        path,
        target_model,
        candidate_name,
        credential_strategy: credential.strategy,
        credential_name: credential.name,
        credential_namespace: credential.namespace,
        credential_key: credential.key,
    })
}

/// Validate selected provider-format fields before request mutation.
fn validate_profile_fields(api_format: &str, revision: &str, path: &str, target_model: &str) -> Result<(), String> {
    if !matches!(api_format, "openai-chat" | "messages") {
        return Err(format!("unsupported selected API format {api_format:?}"));
    }
    if revision.len() != 64
        || !revision
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err("selected provider profile has no valid content-addressed overlay revision".to_owned());
    }
    if !path.starts_with('/')
        || path
            .chars()
            .any(|character| matches!(character, '?' | '#' | '\r' | '\n'))
    {
        return Err("selected provider profile has an invalid request path".to_owned());
    }
    if target_model.trim().is_empty()
        || target_model.len() > 256
        || target_model.bytes().any(|byte| byte.is_ascii_control())
    {
        return Err("selected provider profile has an invalid target model".to_owned());
    }
    Ok(())
}

#[derive(Default)]
/// Secret reference selected with the provider candidate.
struct SelectedCredential {
    /// Credential injection strategy.
    strategy: Option<String>,
    /// Kubernetes Secret name.
    name: Option<String>,
    /// Kubernetes Secret namespace.
    namespace: Option<String>,
    /// Kubernetes Secret key.
    key: Option<String>,
}

/// Read the selected Secret reference from trusted routing metadata.
fn selected_credential(ctx: &HttpFilterContext<'_>) -> SelectedCredential {
    SelectedCredential {
        strategy: ctx.get_metadata(CREDENTIAL_STRATEGY).map(str::to_owned),
        name: ctx.get_metadata(CREDENTIAL_NAME).map(str::to_owned),
        namespace: ctx.get_metadata(CREDENTIAL_NAMESPACE).map(str::to_owned),
        key: ctx.get_metadata(CREDENTIAL_KEY).map(str::to_owned),
    }
}

/// Require the Anthropic credential reference needed for Messages authentication.
fn validate_profile_credential(api_format: &str, credential: &SelectedCredential) -> Result<(), String> {
    if api_format == "messages" {
        if credential.strategy.as_deref() != Some("apikey") {
            return Err("selected Anthropic provider has no supported API-key credential strategy".to_owned());
        }
        for value in [&credential.name, &credential.namespace, &credential.key] {
            if value.as_deref().is_none_or(str::is_empty) {
                return Err("selected Anthropic provider has an incomplete credential reference".to_owned());
            }
        }
    }
    Ok(())
}

/// Read a required non-empty field from trusted routing metadata.
fn required_metadata(ctx: &HttpFilterContext<'_>, key: &str) -> Result<String, String> {
    ctx.get_metadata(key)
        .map(str::to_owned)
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| format!("selected provider profile is missing {key}"))
}

/// Return the profile fields stored under the immutable request-bound prefix.
fn profile_metadata(profile: &SelectedProfile) -> [(&'static str, &str); 10] {
    [
        ("stable_id", profile.stable_id.as_str()),
        ("revision", profile.revision.as_str()),
        ("api_format", profile.api_format.as_str()),
        ("path", profile.path.as_str()),
        ("target_model", profile.target_model.as_str()),
        ("candidate_name", profile.candidate_name.as_str()),
        (
            "credential_strategy",
            profile.credential_strategy.as_deref().unwrap_or(""),
        ),
        ("credential_name", profile.credential_name.as_deref().unwrap_or("")),
        (
            "credential_namespace",
            profile.credential_namespace.as_deref().unwrap_or(""),
        ),
        ("credential_key", profile.credential_key.as_deref().unwrap_or("")),
    ]
}

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::too_many_lines,
    reason = "test assertions and request/response exchanges"
)]
mod tests;
