// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Resolves `file_id` and `file_url` references in `OpenAI` Responses API requests.
//!
//! Walks `message` content arrays and `function_call_output` output
//! arrays, finds content parts that reference files by ID or URL, fetches
//! file metadata and content from an external Files API (e.g. OGX) or
//! remote URL via [`ApiClient`], and inlines the content: raw base64
//! in `file_data` for `input_file`, or a `data:` URL in `image_url` for
//! `input_image`. Forwards configurable headers (`Authorization`,
//! `X-Tenant-ID`) to the Files API for tenant isolation.
//!
//! Praxis runs `StreamBuffer` body hooks before header-phase request
//! filters. Configuration therefore requires an explicit
//! `allow_pre_security_callout: true` acknowledgement and should only
//! be used behind an outer authentication and authorization boundary.
//! Header forwarding is disabled by default; configured headers are
//! copied from the original downstream request.
//!
//! When [`ResponsesState`] is present (e.g. after `rehydrate`),
//! resolved content is synced back into `state.request_body`,
//! `state.messages`, and `state.persisted_messages` so that
//! `responses_proxy` does not overwrite the rewritten body.
//!
//! Content parts with `file_data` or `image_url` pass through unchanged.
//! Content parts with `file_url` are resolved to `file_data` when
//! `file_url: resolve` (default), or passed through when `file_url:
//! passthrough`. No content-part validation — the inference backend
//! handles that.
//!
//! This filter resolves the file transport reference but does not
//! interpret document contents. The inference backend must already
//! support the resulting inline `input_file` / `file_data` or
//! `input_image` / `image_url` part. Backend-specific document
//! adaptation, including extraction to `input_text` for vLLM, is
//! tracked in [#397].
//!
//! `file_id` is supported by the `OpenAI` Responses schema but not
//! baseline `OpenResponses`. This filter intentionally accepts the
//! `OpenAI` extension and converts it to portable inline content.
//!
//! [#397]: https://github.com/praxis-proxy/ai/issues/397
//! [`ApiClient`]: crate::openai::api_client::ApiClient
//! [`ResponsesState`]: super::state::ResponsesState

mod config;
pub(crate) mod resolve;
pub(crate) mod resolve_url;

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::too_many_lines,
    reason = "tests"
)]
mod tests;

use std::borrow::Cow;

use async_trait::async_trait;
use bytes::Bytes;
use praxis_core::callout::{CalloutConfig, FailureMode};
use praxis_filter::{
    BodyAccess, BodyMode, FilterAction, FilterError, HttpFilter, HttpFilterContext, Rejection, parse_filter_config,
};
use tracing::{debug, trace, warn};

use self::{
    config::{FileResolveConfig, FileUrlMode, OnMissing, validate_config},
    resolve::{
        FilesApiClient, FilesApiClientOptions, ResolutionBudget, ResolveError, resolve_input_with_budget, resolve_items,
    },
    resolve_url::{FileUrlResolver, NormalizedOrigin},
};
use super::{openai_responses_proxy::serialized_outbound_body_len, state::ResponsesState};
use crate::{
    classifier::is_responses_create,
    openai::api_client::{ApiClient, ApiClientConfig},
};

/// Resolves `file_id` and `file_url` references in Responses API input
/// by fetching content from a Files API or remote URL via
/// `ApiClient` and inlining the base64-encoded content in the
/// provider-native field.
///
/// The inference backend must support the resulting inline content
/// part. This filter does not extract documents into backend-specific
/// representations such as `input_text`.
///
/// This filter resolves references inside Responses requests; it does
/// not proxy client-facing Files API operations. Route `/v1/files` and
/// its subresources to the configured Files API with the standard
/// `router` and `load_balancer` filters.
///
/// # YAML
///
/// ```yaml
/// filter: openai_file_resolve
/// files_api_url: "http://files-api:8321"
/// allow_private_files_api_url: true
/// allow_pre_security_callout: true
/// ```
///
/// # Full YAML
///
/// ```yaml
/// filter: openai_file_resolve
/// files_api_url: "http://files-api:8321"
/// allow_private_files_api_url: true
/// allow_pre_security_callout: true
/// forward_headers:
///   - authorization
///   - x-tenant-id
/// on_missing: continue
/// timeout_ms: 30000
/// max_body_bytes: 67108864
/// max_file_references: 32
/// ```
///
/// # File URL Resolution YAML
///
/// ```yaml
/// filter: openai_file_resolve
/// files_api_url: "http://ogx:8321"
/// allow_private_files_api_url: true
/// allow_pre_security_callout: true
/// file_url: resolve
/// allowed_file_url_origins:
///   - "https://files.internal:8443"
/// ```
pub struct FileResolveFilter {
    /// Files API HTTP client backed by shared API callout.
    client: FilesApiClient,
    /// Parsed and validated configuration.
    config: FileResolveConfig,
    /// URL resolver for `file_url` references.
    url_resolver: Option<FileUrlResolver>,
}

impl FileResolveFilter {
    /// Create a filter from parsed YAML config.
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] if the YAML config is invalid or the
    /// callout client cannot be constructed.
    #[expect(clippy::too_many_lines, reason = "filter construction boilerplate")]
    pub fn from_config(config: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
        let cfg: FileResolveConfig = parse_filter_config("openai_file_resolve", config)?;
        let validated = validate_config(cfg)?;
        let forward_header_names = prepare_forward_header_names(&validated.forward_headers)?;

        let api_client = ApiClient::new(ApiClientConfig {
            api_base_url: validated.files_api_url.clone(),
            callout_config: CalloutConfig {
                failure_mode: FailureMode::Closed,
                timeout_ms: validated.timeout_ms,
                status_on_error: 502,
                ..CalloutConfig::default()
            },
            forward_header_names,
        })
        .map_err(|e| -> FilterError { format!("openai_file_resolve: {e}").into() })?;

        let client = FilesApiClient::new(
            api_client,
            FilesApiClientOptions {
                max_file_references: validated.max_file_references,
                max_resolved_bytes: validated.max_body_bytes,
            },
        );

        let url_resolver = if validated.file_url == FileUrlMode::Resolve {
            let origins = validated
                .allowed_file_url_origins
                .iter()
                .map(|raw| NormalizedOrigin::parse(raw))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| -> FilterError { format!("openai_file_resolve: {e}").into() })?;
            Some(FileUrlResolver {
                allowed_private_origins: origins,
            })
        } else {
            None
        };

        Ok(Box::new(Self {
            client,
            config: validated,
            url_resolver,
        }))
    }
}

/// Parse header names after configuration validation so outbound
/// requests do not repeat that work.
fn prepare_forward_header_names(names: &[String]) -> Result<Vec<http::HeaderName>, FilterError> {
    names
        .iter()
        .map(|name| name.parse::<http::HeaderName>())
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| format!("openai_file_resolve: failed to prepare forwarded headers: {e}").into())
}

#[async_trait]
impl HttpFilter for FileResolveFilter {
    fn name(&self) -> &'static str {
        "openai_file_resolve"
    }

    fn request_body_access(&self) -> BodyAccess {
        BodyAccess::ReadWrite
    }

    fn request_body_mode(&self) -> BodyMode {
        BodyMode::StreamBuffer {
            max_bytes: Some(self.config.max_body_bytes),
        }
    }

    async fn on_request(&self, _ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        Ok(FilterAction::Continue)
    }

    async fn on_request_body(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        body: &mut Option<Bytes>,
        end_of_stream: bool,
    ) -> Result<FilterAction, FilterError> {
        if !end_of_stream {
            return Ok(FilterAction::Continue);
        }

        if !is_responses_create(&ctx.request.method, ctx.request.uri.path()) {
            trace!("skipping non-create request");
            return Ok(FilterAction::Release);
        }

        if ctx.get_metadata("openai_responses_format.format") != Some("openai_responses") {
            trace!("skipping non-responses request");
            return Ok(FilterAction::Release);
        }

        let Some(raw) = body.as_ref() else {
            trace!("no body, releasing");
            return Ok(FilterAction::Release);
        };
        if raw.len() > self.config.max_body_bytes {
            return Ok(reject_raw_body_too_large(raw.len(), self.config.max_body_bytes));
        }

        let mut parsed: serde_json::Value = match serde_json::from_slice(raw) {
            Ok(v) => v,
            Err(e) => {
                debug!(error = %e, "body is not valid JSON, releasing");
                return Ok(FilterAction::Release);
            },
        };

        resolve_and_rewrite(self, ctx, body, &mut parsed).await
    }
}

/// Run resolution and rewrite the body if any references were resolved.
async fn resolve_and_rewrite(
    filter: &FileResolveFilter,
    ctx: &mut HttpFilterContext<'_>,
    body: &mut Option<Bytes>,
    parsed: &mut serde_json::Value,
) -> Result<FilterAction, FilterError> {
    let mut budget = filter.client.resolution_budget();
    let count = match resolve_current_input(filter, ctx, parsed, &mut budget).await {
        Ok(count) => count,
        Err(e) => return Ok(reject_resolve_error(&e)),
    };
    if count == 0 {
        trace!("no file_id references found");
        if let Err(e) = update_state(filter, ctx, None, &mut budget).await {
            return Ok(reject_resolve_error(&e));
        }
        if let Some(rejection) = reject_oversized_state_body(ctx, filter.config.max_body_bytes)? {
            return Ok(rejection);
        }
        return Ok(FilterAction::Continue);
    }

    debug!(count, "resolved file_id references");
    if let Some(rejection) = rewrite_body(body, parsed, ctx, filter.config.max_body_bytes)? {
        return Ok(rejection);
    }
    if let Err(e) = update_state(filter, ctx, Some(parsed), &mut budget).await {
        return Ok(reject_resolve_error(&e));
    }
    if let Some(rejection) = reject_oversized_state_body(ctx, filter.config.max_body_bytes)? {
        return Ok(rejection);
    }

    Ok(FilterAction::Continue)
}

/// Enforce the resolver's body limit against the exact request shape
/// that `openai_responses_proxy` will later serialize from state.
fn reject_oversized_state_body(
    ctx: &HttpFilterContext<'_>,
    max_body_bytes: usize,
) -> Result<Option<FilterAction>, FilterError> {
    let Some(state) = ctx.extensions.get::<ResponsesState>() else {
        return Ok(None);
    };
    let len = serialized_outbound_body_len(state).map_err(|e| -> FilterError {
        format!("openai_file_resolve: failed to measure rebuilt request body: {e}").into()
    })?;
    Ok((len > max_body_bytes).then(|| reject_rewritten_body_too_large(len, max_body_bytes)))
}

/// Resolve references in the request body's current input.
async fn resolve_current_input(
    filter: &FileResolveFilter,
    ctx: &HttpFilterContext<'_>,
    parsed: &mut serde_json::Value,
    budget: &mut ResolutionBudget,
) -> Result<usize, ResolveError> {
    Box::pin(resolve_input_with_budget(
        parsed,
        &filter.client,
        filter.config.on_missing,
        &ctx.request.headers,
        filter.url_resolver.as_ref(),
        budget,
    ))
    .await
}

/// Resolve history and synchronize state after the body walk.
async fn update_state(
    filter: &FileResolveFilter,
    ctx: &mut HttpFilterContext<'_>,
    resolved_body: Option<&serde_json::Value>,
    budget: &mut ResolutionBudget,
) -> Result<(), ResolveError> {
    match resolved_body {
        Some(body) => {
            Box::pin(sync_state_with_budget(
                ctx,
                body,
                &filter.client,
                filter.config.on_missing,
                filter.url_resolver.as_ref(),
                budget,
            ))
            .await
        },
        None => {
            Box::pin(resolve_state_history(
                ctx,
                &filter.client,
                filter.config.on_missing,
                filter.url_resolver.as_ref(),
                budget,
            ))
            .await
        },
    }
}

/// Serialize the resolved JSON and replace the buffered request body.
fn rewrite_body(
    body: &mut Option<Bytes>,
    parsed: &serde_json::Value,
    ctx: &mut HttpFilterContext<'_>,
    max_body_bytes: usize,
) -> Result<Option<FilterAction>, FilterError> {
    let rewritten = serde_json::to_vec(parsed)
        .map_err(|e| -> FilterError { format!("openai_file_resolve: failed to serialize body: {e}").into() })?;
    let len = rewritten.len();
    if len > max_body_bytes {
        return Ok(Some(reject_rewritten_body_too_large(len, max_body_bytes)));
    }
    *body = Some(Bytes::from(rewritten));
    ctx.extra_request_headers
        .push((Cow::Borrowed("content-length"), len.to_string()));
    Ok(None)
}

/// Sync resolved content back into [`ResponsesState`] so that
/// `responses_proxy` does not overwrite the rewritten body with
/// stale data when it rebuilds from state.
///
/// Updates `request_body`, and replaces the current-input tail of
/// `messages` / `persisted_messages` with the resolved items.
/// History messages prepended by rehydrate are also walked so
/// that any `file_id` references in them are resolved.
#[expect(clippy::too_many_arguments, reason = "threading resolver through state sync")]
async fn sync_state_with_budget(
    ctx: &mut HttpFilterContext<'_>,
    resolved_body: &serde_json::Value,
    client: &FilesApiClient,
    on_missing: OnMissing,
    url_resolver: Option<&FileUrlResolver>,
    budget: &mut ResolutionBudget,
) -> Result<(), ResolveError> {
    let Some(state) = ctx.extensions.get_mut::<ResponsesState>() else {
        return Ok(());
    };

    state.request_body = resolved_body.clone();

    let Some(resolved_input) = resolved_body.get("input").and_then(serde_json::Value::as_array) else {
        return Ok(());
    };

    let input_len = state.input.len();
    let resolver = HistoryResolver {
        client,
        on_missing,
        request_headers: &ctx.request.headers,
        url_resolver,
    };

    sync_message_history(&mut state.messages, input_len, Some(resolved_input), resolver, budget).await?;
    sync_persisted_history(
        &mut state.persisted_messages,
        input_len,
        Some(resolved_input),
        resolver,
        budget,
    )
    .await
}

/// Test helper that creates an isolated request resolution budget.
#[cfg(test)]
async fn sync_state(
    ctx: &mut HttpFilterContext<'_>,
    resolved_body: &serde_json::Value,
    client: &FilesApiClient,
    on_missing: OnMissing,
) -> Result<(), ResolveError> {
    let mut budget = client.resolution_budget();
    sync_state_with_budget(ctx, resolved_body, client, on_missing, None, &mut budget).await
}

/// Resolve file references in rehydrated history when the
/// current input did not require a body rewrite.
async fn resolve_state_history(
    ctx: &mut HttpFilterContext<'_>,
    client: &FilesApiClient,
    on_missing: OnMissing,
    url_resolver: Option<&FileUrlResolver>,
    budget: &mut ResolutionBudget,
) -> Result<(), ResolveError> {
    let Some(state) = ctx.extensions.get_mut::<ResponsesState>() else {
        return Ok(());
    };

    let input_len = state.input.len();
    let resolver = HistoryResolver {
        client,
        on_missing,
        request_headers: &ctx.request.headers,
        url_resolver,
    };

    sync_message_history(&mut state.messages, input_len, None, resolver, budget).await?;
    sync_persisted_history(&mut state.persisted_messages, input_len, None, resolver, budget).await
}

/// Shared dependencies for resolving one state history vector.
#[derive(Clone, Copy)]
struct HistoryResolver<'a> {
    /// Files API client used for history references.
    client: &'a FilesApiClient,
    /// Configured behavior when a history reference cannot resolve.
    on_missing: OnMissing,
    /// Original request headers available for configured forwarding.
    request_headers: &'a http::HeaderMap,
    /// URL resolver for `file_url` references.
    url_resolver: Option<&'a FileUrlResolver>,
}

/// Resolve the persistence mirror with independent count and byte
/// accounting while reusing the request-wide cache and deadline.
async fn sync_persisted_history(
    messages: &mut [serde_json::Value],
    input_len: usize,
    resolved_input: Option<&[serde_json::Value]>,
    resolver: HistoryResolver<'_>,
    budget: &mut ResolutionBudget,
) -> Result<(), ResolveError> {
    let saved = budget.begin_independent_accounting();
    let result = sync_message_history(messages, input_len, resolved_input, resolver, budget).await;
    budget.restore_accounting(saved);
    result
}

/// Update the current-input tail, when provided, then resolve the
/// independently sized history prefix.
async fn sync_message_history(
    messages: &mut [serde_json::Value],
    input_len: usize,
    resolved_input: Option<&[serde_json::Value]>,
    resolver: HistoryResolver<'_>,
    budget: &mut ResolutionBudget,
) -> Result<(), ResolveError> {
    let Some(history_end) = messages.len().checked_sub(input_len) else {
        return Ok(());
    };
    if let Some(resolved_input) = resolved_input {
        replace_tail(messages, history_end, resolved_input);
    }
    resolve_history(messages, history_end, resolver, budget).await
}

/// Copy resolved input items into the current-input tail of a
/// message vector, starting at `history_end`.
fn replace_tail(messages: &mut [serde_json::Value], history_end: usize, resolved_input: &[serde_json::Value]) {
    for (i, item) in resolved_input.iter().enumerate() {
        if let Some(slot) = messages.get_mut(history_end + i) {
            *slot = item.clone();
        }
    }
}

/// Resolve file references in history messages (the prefix
/// before the current input).
async fn resolve_history(
    messages: &mut [serde_json::Value],
    history_end: usize,
    resolver: HistoryResolver<'_>,
    budget: &mut ResolutionBudget,
) -> Result<(), ResolveError> {
    if history_end == 0 {
        return Ok(());
    }
    let Some(history) = messages.get_mut(..history_end) else {
        return Ok(());
    };
    resolve_items(
        history,
        resolver.client,
        resolver.on_missing,
        resolver.request_headers,
        resolver.url_resolver,
        budget,
    )
    .await
    .map(|_count| ())
}

/// Map one resolution error to an HTTP status and safe client message.
fn resolve_error_response(err: &ResolveError) -> (u16, String) {
    match err {
        ResolveError::CalloutFailed { file_id, detail } => callout_error_response(file_id, detail),
        ResolveError::InvalidFileId { file_id, detail } => invalid_id_error_response(file_id, detail),
        ResolveError::TooManyReferences { limit } => too_many_error_response(*limit),
        ResolveError::TooLarge { reference, limit } => too_large_error_response(reference, *limit),
        ResolveError::FileUrlBlocked { label } => file_url_blocked_response(label),
        ResolveError::FileUrlFailed { label, detail } => file_url_failed_response(label, detail),
    }
}

/// Report a Files API failure to the caller.
fn callout_error_response(file_id: &str, detail: &str) -> (u16, String) {
    warn!(file_id, detail, "callout failed during file resolution");
    (
        502,
        format!("failed to resolve file '{file_id}': Files API request failed"),
    )
}

/// Report an invalid file ID to the caller.
fn invalid_id_error_response(file_id: &str, detail: &str) -> (u16, String) {
    warn!(file_id, detail, "invalid file id during file resolution");
    (400, format!("failed to resolve file '{file_id}': {detail}"))
}

/// Report an exceeded reference-count cap to the caller.
fn too_many_error_response(limit: usize) -> (u16, String) {
    warn!(limit, "request exceeds file reference limit");
    (413, format!("request exceeds {limit} file references"))
}

/// Report an exceeded resolved-body size cap to the caller.
fn too_large_error_response(reference: &str, limit: usize) -> (u16, String) {
    warn!(reference, limit, "resolved file exceeds configured limit");
    (
        413,
        format!("failed to resolve file reference '{reference}': resolved content exceeds {limit} bytes"),
    )
}

/// Report a file URL blocked by SSRF policy to the caller.
fn file_url_blocked_response(label: &str) -> (u16, String) {
    warn!(url = %label, "file URL blocked by security policy");
    (403, format!("file URL '{label}' blocked by security policy"))
}

/// Report a file URL fetch failure to the caller.
fn file_url_failed_response(label: &str, detail: &str) -> (u16, String) {
    warn!(url = %label, detail, "file URL fetch failed");
    (502, format!("failed to fetch file URL '{label}': request failed"))
}

/// Build a rejection response from a resolution error.
fn reject_resolve_error(err: &ResolveError) -> FilterAction {
    let (status, message) = resolve_error_response(err);

    let body = serde_json::json!({
        "error": {
            "message": message,
            "type": "file_resolve_error"
        }
    })
    .to_string();

    FilterAction::Reject(
        Rejection::status(status)
            .with_header("content-type", "application/json")
            .with_body(Bytes::from(body)),
    )
}

/// Build a rejection before parsing or resolving an oversized raw body.
fn reject_raw_body_too_large(actual: usize, limit: usize) -> FilterAction {
    warn!(actual, limit, "buffered request body exceeds configured limit");
    let body = serde_json::json!({
        "error": {
            "message": format!("request body exceeds {limit} bytes"),
            "type": "file_resolve_error"
        }
    })
    .to_string();

    FilterAction::Reject(
        Rejection::status(413)
            .with_header("content-type", "application/json")
            .with_body(Bytes::from(body)),
    )
}

/// Build a rejection response when the final rewritten JSON body
/// is too large.
fn reject_rewritten_body_too_large(actual: usize, limit: usize) -> FilterAction {
    warn!(actual, limit, "rewritten request body exceeds configured limit");
    let body = serde_json::json!({
        "error": {
            "message": format!("resolved request body exceeds {limit} bytes"),
            "type": "file_resolve_error"
        }
    })
    .to_string();

    FilterAction::Reject(
        Rejection::status(413)
            .with_header("content-type", "application/json")
            .with_body(Bytes::from(body)),
    )
}
