// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Proto-to-Praxis conversions for `ext_proc` header mutations.
//!
//! Translates between the Envoy `ext_proc` protobuf types and
//! Praxis filter context operations: building [`HttpHeaders`]
//! from request/response state and applying [`HeaderMutation`]
//! and [`ImmediateResponse`] results back to the context.
//!
//! [`HttpHeaders`]: crate::proto::envoy::service::ext_proc::v3::HttpHeaders
//! [`HeaderMutation`]: crate::proto::envoy::service::ext_proc::v3::HeaderMutation
//! [`ImmediateResponse`]: crate::proto::envoy::service::ext_proc::v3::ImmediateResponse

use std::borrow::Cow;

use bytes::Bytes;
use praxis_filter::{FilterAction, HttpFilterContext, Rejection, TrustedHeaderMutation};

use crate::{
    Phase,
    proto::envoy::service::{
        common::v3::{HeaderValue, HeaderValueOption, header_value_option::HeaderAppendAction},
        ext_proc::v3::{HeaderMap, HeaderMutation, HeadersResponse, HttpHeaders, ImmediateResponse},
    },
};

/// Reserved header name on the EPP response.
const EPP_DESTINATION_NAME: &str = "x-gateway-destination-endpoint";
/// Typed key for the reserved destination in Core's mutation queues.
const EPP_DESTINATION: http::HeaderName = http::HeaderName::from_static(EPP_DESTINATION_NAME);

/// Deferred llm-d callouts carry the routing contract and request correlation.
/// Credentials from ingress or gateway policy can use arbitrary header names.
const EPP_REQUEST_HEADERS: &[&str] = &[
    "content-type",
    "host",
    "traceparent",
    "tracestate",
    "x-request-id",
    "x-gateway-model-name",
    "x-gateway-base-model-name",
    "x-llm-d-inference-objective",
    "x-gateway-inference-objective",
    "x-llm-d-model-name-rewrite",
    "x-gateway-model-name-rewrite",
];

// -----------------------------------------------------------------------------
// Request -> Proto
// -----------------------------------------------------------------------------

/// Build [`HttpHeaders`] from the current request context.
///
/// Includes `:method`, `:path`, `:scheme`, and `:authority`
/// pseudo-headers followed by all request headers, matching
/// the Envoy `ext_proc` convention that external processors
/// expect.
pub(crate) fn request_to_proto_headers(ctx: &HttpFilterContext<'_>) -> HttpHeaders {
    serialize_request_headers(ctx, &ctx.request.headers, false)
}

/// Start a picker exchange with no destination inherited from before the call.
pub(crate) fn prepare_deferred_request(ctx: &mut HttpFilterContext<'_>) -> HttpHeaders {
    let headers = effective_request_to_proto_headers(ctx);
    clear_picker_destination(ctx);
    headers
}

/// Project the post-policy request onto the picker header contract.
fn effective_request_to_proto_headers(ctx: &HttpFilterContext<'_>) -> HttpHeaders {
    if ctx.request_headers_to_remove.is_empty()
        && ctx.request_headers_to_set.is_empty()
        && ctx.request_header_mutations.is_empty()
    {
        return serialize_request_headers(ctx, &ctx.request.headers, true);
    }

    // Core already applied the pre-read log. Replay policy edits and leave
    // extra_request_headers on the worker hop. That queue holds egress
    // additions, including credentials scoped to the selected worker cluster.
    let mut headers = ctx.request.headers.clone();
    for name in &ctx.request_headers_to_remove {
        headers.remove(name);
    }
    for (name, value) in &ctx.request_headers_to_set {
        headers.insert(name.clone(), value.clone());
    }
    for mutation in &ctx.request_header_mutations {
        mutation.apply_to(&mut headers);
    }
    serialize_request_headers(ctx, &headers, true)
}

/// Clear earlier edits so only this EPP exchange can publish a destination.
fn clear_picker_destination(ctx: &mut HttpFilterContext<'_>) {
    ctx.request_headers_to_set.retain(|(name, _)| *name != EPP_DESTINATION);
    ctx.extra_request_headers
        .retain(|(name, _)| !name.eq_ignore_ascii_case(EPP_DESTINATION_NAME));
    ctx.request_header_mutations
        .retain(|edit| !edit.matches_header(&EPP_DESTINATION));
    ctx.pre_read_mutations
        .retain(|edit| !edit.matches_header(&EPP_DESTINATION));
    if !ctx.request_headers_to_remove.contains(&EPP_DESTINATION) {
        ctx.request_headers_to_remove.push(EPP_DESTINATION);
    }
}

/// Encode pseudo-headers and the chosen request header view for gRPC.
fn serialize_request_headers(
    ctx: &HttpFilterContext<'_>,
    request_headers: &http::HeaderMap,
    epp_only: bool,
) -> HttpHeaders {
    let path = ctx
        .request
        .uri
        .path_and_query()
        .map_or_else(|| ctx.request.uri.path(), http::uri::PathAndQuery::as_str);
    let scheme = if ctx.downstream_tls { "https" } else { "http" };

    let mut headers = vec![
        proto_header(":method", ctx.request.method.as_str()),
        proto_header(":path", path),
        proto_header(":scheme", scheme),
    ];

    if let Some(authority) = request_headers.get(http::header::HOST) {
        headers.push(proto_header_value(":authority", authority));
    }

    for (name, value) in request_headers {
        if epp_only && !EPP_REQUEST_HEADERS.contains(&name.as_str()) {
            continue;
        }
        headers.push(proto_header_value(name.as_str(), value));
    }
    if epp_only
        && request_headers.contains_key(http::header::CONTENT_LENGTH)
        && let Some(body) = &ctx.buffered_request_body
    {
        headers.push(proto_header("content-length", &body.len().to_string()));
    }

    HttpHeaders {
        headers: Some(HeaderMap { headers }),
        end_of_stream: false,
    }
}

/// Build [`HttpHeaders`] from the upstream response context.
///
/// Includes a `:status` pseudo-header followed by all response
/// headers. Returns empty headers when `ctx.response_header` is
/// `None` (should not happen during the response phase).
pub(crate) fn response_to_proto_headers(ctx: &HttpFilterContext<'_>) -> HttpHeaders {
    let mut headers = Vec::new();

    if let Some(resp) = ctx.response_header.as_ref() {
        headers.push(proto_header(":status", &resp.status.as_u16().to_string()));

        for (name, value) in &resp.headers {
            headers.push(proto_header_value(name.as_str(), value));
        }
    }

    HttpHeaders {
        headers: Some(HeaderMap { headers }),
        end_of_stream: false,
    }
}

// -----------------------------------------------------------------------------
// Proto -> Praxis mutations
// -----------------------------------------------------------------------------

/// Apply a [`HeadersResponse`] to the filter context.
///
/// Delegates to request or response mutation based on the
/// current processing [`Phase`].
pub(crate) fn apply_headers_response(hr: &HeadersResponse, ctx: &mut HttpFilterContext<'_>, phase: Phase) {
    let Some(common) = &hr.response else {
        return;
    };
    let Some(mutation) = &common.header_mutation else {
        return;
    };

    match phase {
        Phase::Request => apply_request_header_mutation(mutation, ctx),
        Phase::Response => apply_response_header_mutation(mutation, ctx),
    }
}

/// Apply header mutations to the upstream request.
///
/// Maps each [`HeaderAppendAction`] variant to the appropriate
/// context queue:
///
/// - `AppendIfExistsOrAdd` (default) -> [`extra_request_headers`]
/// - `OverwriteIfExistsOrAdd` -> [`request_headers_to_set`]
/// - `OverwriteIfExists` -> [`request_headers_to_set`] (only if present)
/// - `AddIfAbsent` -> [`extra_request_headers`] (only if absent)
///
/// Pseudo-headers (`:` prefix) are skipped because Praxis sets
/// method, path, scheme, and authority through dedicated fields.
///
/// [`HeaderAppendAction`]: crate::proto::envoy::service::common::v3::header_value_option::HeaderAppendAction
/// [`extra_request_headers`]: HttpFilterContext::extra_request_headers
/// [`request_headers_to_set`]: HttpFilterContext::request_headers_to_set
pub(crate) fn apply_request_header_mutation(mutation: &HeaderMutation, ctx: &mut HttpFilterContext<'_>) {
    apply_request_mutation(mutation, ctx, false);
}

/// Apply EPP edits after earlier request-phase policy has run.
pub(crate) fn apply_deferred_request_header_mutation(mutation: &HeaderMutation, ctx: &mut HttpFilterContext<'_>) {
    apply_request_mutation(mutation, ctx, true);
}

/// Process removals before sets in the exchange's request phase.
fn apply_request_mutation(mutation: &HeaderMutation, ctx: &mut HttpFilterContext<'_>, deferred: bool) {
    remove_request_headers(&mutation.remove_headers, ctx, deferred);
    set_request_headers(&mutation.set_headers, ctx, deferred);
}

/// Queue request header removals, skipping pseudo-headers and `Host`.
fn remove_request_headers(names: &[String], ctx: &mut HttpFilterContext<'_>, deferred: bool) {
    for name in names {
        if is_pseudo_header(name) || is_request_authority(name) {
            continue;
        }
        if let Ok(header_name) = http::HeaderName::try_from(name.as_str()) {
            if deferred {
                if header_name == EPP_DESTINATION {
                    clear_picker_destination(ctx);
                } else {
                    ctx.request_header_mutations
                        .push(TrustedHeaderMutation::Remove(header_name));
                }
            } else {
                begin_ordered_pre_read_mutations(ctx);
                ctx.request_headers_to_remove.push(header_name.clone());
                ctx.pre_read_mutations.push(TrustedHeaderMutation::Remove(header_name));
            }
        }
    }
}

/// Apply set-header mutations to the request context.
fn set_request_headers(headers: &[HeaderValueOption], ctx: &mut HttpFilterContext<'_>, deferred: bool) {
    for hvo in headers {
        let Some(hv) = &hvo.header else { continue };
        if is_pseudo_header(&hv.key) || is_request_authority(&hv.key) {
            continue;
        }
        let Ok(name) = http::HeaderName::try_from(hv.key.as_str()) else {
            continue;
        };
        if deferred {
            if name == EPP_DESTINATION {
                apply_picker_destination(hvo, hv, ctx);
            } else {
                apply_deferred_request_header(hvo, hv, name, ctx);
            }
        } else {
            dispatch_request_header(hvo, hv, name, ctx);
        }
    }
}

/// Resolve a conditional reply edit before worker-only header injections run.
fn effective_request_has_header(ctx: &HttpFilterContext<'_>, name: &http::HeaderName) -> bool {
    if let Some(edit) = ctx
        .request_header_mutations
        .iter()
        .rev()
        .find(|edit| edit.matches_header(name))
    {
        return !matches!(edit, TrustedHeaderMutation::Remove(_));
    }
    ctx.request_headers_to_set.iter().any(|(header, _)| header == name)
        || (ctx.request.headers.contains_key(name) && !ctx.request_headers_to_remove.contains(name))
}

/// Apply the reply after earlier policy, preserving replacement on worker
/// writeback for the append action previously routed through the extras queue.
fn apply_deferred_request_header(
    hvo: &HeaderValueOption,
    hv: &HeaderValue,
    name: http::HeaderName,
    ctx: &mut HttpFilterContext<'_>,
) {
    let action = resolve_append_action(hvo);
    if (action == HeaderAppendAction::OverwriteIfExists && !effective_request_has_header(ctx, &name))
        || (action == HeaderAppendAction::AddIfAbsent && effective_request_has_header(ctx, &name))
    {
        return;
    }
    if let Ok(value) = http::HeaderValue::try_from(header_value_string(hv)) {
        ctx.request_header_mutations
            .push(TrustedHeaderMutation::Set(name, value));
    }
}

/// Publish the reserved destination through the existing selector's queues.
fn apply_picker_destination(hvo: &HeaderValueOption, hv: &HeaderValue, ctx: &mut HttpFilterContext<'_>) {
    // The deferred exchange starts with no destination. Conditional actions
    // can see earlier picker replies, never the original client header.
    let exists = ctx
        .request_headers_to_set
        .iter()
        .any(|(name, _)| *name == EPP_DESTINATION)
        || ctx
            .extra_request_headers
            .iter()
            .any(|(name, _)| name.eq_ignore_ascii_case(EPP_DESTINATION_NAME));
    let action = resolve_append_action(hvo);
    let value = header_value_string(hv);
    let Ok(header_value) = http::HeaderValue::from_str(&value) else {
        return;
    };
    match action {
        HeaderAppendAction::OverwriteIfExists if !exists => (),
        HeaderAppendAction::AddIfAbsent if exists => (),
        HeaderAppendAction::OverwriteIfExists | HeaderAppendAction::OverwriteIfExistsOrAdd => {
            clear_picker_destination(ctx);
            ctx.request_headers_to_set.push((EPP_DESTINATION, header_value));
        },
        HeaderAppendAction::AppendIfExistsOrAdd | HeaderAppendAction::AddIfAbsent => {
            ctx.extra_request_headers
                .push((Cow::Borrowed(EPP_DESTINATION_NAME), value));
        },
    }
}

/// Route a single request header mutation to the correct context queue.
#[expect(
    clippy::too_many_lines,
    reason = "append-action variants each preserve ordered pre-read provenance"
)]
fn dispatch_request_header(
    hvo: &HeaderValueOption,
    hv: &HeaderValue,
    name: http::HeaderName,
    ctx: &mut HttpFilterContext<'_>,
) {
    use crate::proto::envoy::service::common::v3::header_value_option::HeaderAppendAction;

    let value = header_value_string(hv);

    match resolve_append_action(hvo) {
        HeaderAppendAction::OverwriteIfExistsOrAdd => {
            if let Ok(v) = http::HeaderValue::try_from(&value) {
                begin_ordered_pre_read_mutations(ctx);
                ctx.request_headers_to_set.push((name.clone(), v.clone()));
                ctx.pre_read_mutations.push(TrustedHeaderMutation::Set(name, v));
            }
        },
        HeaderAppendAction::OverwriteIfExists => {
            if ctx.request.headers.contains_key(&name)
                && let Ok(v) = http::HeaderValue::try_from(&value)
            {
                begin_ordered_pre_read_mutations(ctx);
                ctx.request_headers_to_set.push((name.clone(), v.clone()));
                ctx.pre_read_mutations.push(TrustedHeaderMutation::Set(name, v));
            }
        },
        HeaderAppendAction::AddIfAbsent => {
            if !ctx.request.headers.contains_key(&name) {
                begin_ordered_pre_read_mutations(ctx);
                ctx.extra_request_headers
                    .push((Cow::Owned(hv.key.clone()), value.clone()));
                ctx.pre_read_mutations.push(TrustedHeaderMutation::Add(name, value));
            }
        },
        HeaderAppendAction::AppendIfExistsOrAdd => {
            begin_ordered_pre_read_mutations(ctx);
            ctx.extra_request_headers
                .push((Cow::Owned(hv.key.clone()), value.clone()));
            ctx.pre_read_mutations.push(TrustedHeaderMutation::Add(name, value));
        },
    }
}

/// Preserve earlier legacy body mutations before activating ordered mode.
///
/// Core treats a non-empty ordered log as authoritative for the entire
/// pre-read pass. Seed it from the grouped queues before the first ext-proc
/// mutation so filters that ran earlier are not silently discarded.
///
/// This does not make later legacy producers composable with ordered mode;
/// ext-proc already activated that mode before this bridge existed. Unifying
/// the two mutation APIs is owned by Praxis Core issue #1072. This bridge is
/// deliberately limited to retaining mutations that are knowable when
/// ext-proc crosses the existing ordered-mode boundary.
pub(crate) fn begin_ordered_pre_read_mutations(ctx: &mut HttpFilterContext<'_>) {
    if !ctx.pre_read_mutations.is_empty() {
        return;
    }
    ctx.pre_read_mutations.extend(
        ctx.request_headers_to_remove
            .iter()
            .cloned()
            .map(TrustedHeaderMutation::Remove),
    );
    ctx.pre_read_mutations.extend(
        ctx.request_headers_to_set
            .iter()
            .cloned()
            .map(|(name, value)| TrustedHeaderMutation::Set(name, value)),
    );
    for (name, value) in &ctx.extra_request_headers {
        if let (Ok(name), Ok(_value)) = (
            http::HeaderName::from_bytes(name.as_bytes()),
            http::HeaderValue::from_str(value),
        ) {
            ctx.pre_read_mutations
                .push(TrustedHeaderMutation::Add(name, value.clone()));
        }
    }
}

/// Apply header mutations to the upstream response.
///
/// Modifies [`HttpFilterContext::response_header`] directly and
/// sets [`HttpFilterContext::response_headers_modified`] when
/// any mutation is applied. Pseudo-headers are skipped.
pub(crate) fn apply_response_header_mutation(mutation: &HeaderMutation, ctx: &mut HttpFilterContext<'_>) {
    let Some(resp) = ctx.response_header.as_mut() else {
        return;
    };

    let removes = remove_response_headers(&mutation.remove_headers, resp);
    let sets = set_response_headers(&mutation.set_headers, resp);

    if sets || removes {
        ctx.response_headers_modified = true;
    }
}

/// Apply set-header mutations to a response, returning whether any were applied.
fn set_response_headers(headers: &[HeaderValueOption], resp: &mut praxis_filter::Response) -> bool {
    let mut modified = false;
    for hvo in headers {
        let Some(hv) = &hvo.header else { continue };
        if is_pseudo_header(&hv.key) {
            continue;
        }
        let Ok(name) = http::HeaderName::try_from(hv.key.as_str()) else {
            continue;
        };
        let value = header_value_string(hv);
        let Ok(val) = http::HeaderValue::try_from(&value) else {
            continue;
        };

        if dispatch_response_header(hvo, name, val, resp) {
            modified = true;
        }
    }
    modified
}

/// Route a single response header mutation, returning whether it was applied.
fn dispatch_response_header(
    hvo: &HeaderValueOption,
    name: http::HeaderName,
    val: http::HeaderValue,
    resp: &mut praxis_filter::Response,
) -> bool {
    use crate::proto::envoy::service::common::v3::header_value_option::HeaderAppendAction;

    match resolve_append_action(hvo) {
        HeaderAppendAction::AppendIfExistsOrAdd => {
            resp.headers.append(name, val);
            true
        },
        HeaderAppendAction::OverwriteIfExistsOrAdd => {
            resp.headers.insert(name, val);
            true
        },
        HeaderAppendAction::OverwriteIfExists => {
            if resp.headers.contains_key(&name) {
                resp.headers.insert(name, val);
                true
            } else {
                false
            }
        },
        HeaderAppendAction::AddIfAbsent => {
            if resp.headers.contains_key(&name) {
                false
            } else {
                resp.headers.append(name, val);
                true
            }
        },
    }
}

/// Apply remove-header mutations to a response, returning whether any were applied.
fn remove_response_headers(names: &[String], resp: &mut praxis_filter::Response) -> bool {
    let mut modified = false;
    for name in names {
        if is_pseudo_header(name) {
            continue;
        }
        if let Ok(header_name) = http::HeaderName::try_from(name.as_str())
            && resp.headers.remove(&header_name).is_some()
        {
            modified = true;
        }
    }
    modified
}

/// Convert an [`ImmediateResponse`] to a [`FilterAction::Reject`].
///
/// Maps the proto status code (defaulting to 200 when absent),
/// body, and response headers to a [`Rejection`].
pub(crate) fn immediate_to_rejection(imm: &ImmediateResponse) -> FilterAction {
    let status = imm.status.as_ref().map_or(200, |s| {
        let code = s.code;
        u16::try_from(code).unwrap_or(500)
    });

    let status = if (100..=599).contains(&status) { status } else { 500 };

    let mut rejection = Rejection::status(status);

    if !imm.body.is_empty() {
        rejection = rejection.with_body(Bytes::copy_from_slice(imm.body.as_bytes()));
    }

    if let Some(hm) = &imm.headers {
        for hvo in &hm.set_headers {
            if let Some(hv) = &hvo.header {
                let value = header_value_string(hv);
                rejection = rejection.with_header(hv.key.clone(), value);
            }
        }
    }

    FilterAction::Reject(rejection)
}

// -----------------------------------------------------------------------------
// Utilities
// -----------------------------------------------------------------------------

/// Extract the string value from a [`HeaderValue`].
///
/// Prefers `raw_value` (as UTF-8) over `value` when non-empty,
/// matching the Envoy convention where `raw_value` carries the
/// original bytes.
pub(crate) fn header_value_string(hv: &HeaderValue) -> String {
    if hv.raw_value.is_empty() {
        hv.value.clone()
    } else {
        String::from_utf8_lossy(&hv.raw_value).into_owned()
    }
}

/// Returns `true` if the header name is an HTTP/2 pseudo-header.
pub(crate) fn is_pseudo_header(name: &str) -> bool {
    name.starts_with(':')
}

/// Returns `true` if the header is protocol-controlled request authority.
///
/// `Host` is the HTTP/1.1 singleton authority header. Applying a default
/// `AppendIfExistsOrAdd` `ext_proc` mutation would create duplicate `Host`
/// fields on the forwarded request, so request mutations never alter it.
fn is_request_authority(name: &str) -> bool {
    name.eq_ignore_ascii_case("host")
}

/// Build a [`HeaderValue`] proto with the given key and value.
fn proto_header(key: &str, value: &str) -> HeaderValue {
    HeaderValue {
        key: key.to_owned(),
        value: value.to_owned(),
        raw_value: Vec::new(),
    }
}

/// Convert an HTTP header value while preserving bytes that are not valid UTF-8.
fn proto_header_value(key: &str, value: &http::HeaderValue) -> HeaderValue {
    match value.to_str() {
        Ok(value) => proto_header(key, value),
        Err(_) => HeaderValue {
            key: key.to_owned(),
            value: String::new(),
            raw_value: value.as_bytes().to_vec(),
        },
    }
}

/// Resolve the [`HeaderAppendAction`] for a [`HeaderValueOption`].
///
/// Uses `append_action` when set (non-zero). Falls back to the
/// deprecated `append` field, mapping `true` / default to
/// [`AppendIfExistsOrAdd`] and `false` to
/// [`OverwriteIfExistsOrAdd`], matching proto3 default semantics.
///
/// [`HeaderAppendAction`]: crate::proto::envoy::service::common::v3::header_value_option::HeaderAppendAction
/// [`AppendIfExistsOrAdd`]: crate::proto::envoy::service::common::v3::header_value_option::HeaderAppendAction::AppendIfExistsOrAdd
/// [`OverwriteIfExistsOrAdd`]: crate::proto::envoy::service::common::v3::header_value_option::HeaderAppendAction::OverwriteIfExistsOrAdd
fn resolve_append_action(hvo: &HeaderValueOption) -> HeaderAppendAction {
    if hvo.append_action != 0 {
        return HeaderAppendAction::try_from(hvo.append_action).unwrap_or(HeaderAppendAction::AppendIfExistsOrAdd);
    }

    // proto3 default for append_action is 0 (APPEND_IF_EXISTS_OR_ADD).
    // Fall back to deprecated `append`; default to true (append)
    // when neither field is explicitly set.
    if hvo.append.unwrap_or(true) {
        HeaderAppendAction::AppendIfExistsOrAdd
    } else {
        HeaderAppendAction::OverwriteIfExistsOrAdd
    }
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::unwrap_used, clippy::expect_used, reason = "tests")]
mod tests {
    use super::*;
    use crate::proto::envoy::service::common::v3::{HttpStatus, header_value_option::HeaderAppendAction};

    // -------------------------------------------------------------------------
    // header_value_string
    // -------------------------------------------------------------------------

    #[test]
    fn header_value_string_prefers_raw_value() {
        let hv = HeaderValue {
            key: "x-test".to_owned(),
            value: "fallback".to_owned(),
            raw_value: b"primary".to_vec(),
        };
        assert_eq!(
            header_value_string(&hv),
            "primary",
            "should prefer raw_value when non-empty"
        );
    }

    #[test]
    fn header_value_string_falls_back_to_value() {
        let hv = HeaderValue {
            key: "x-test".to_owned(),
            value: "fallback".to_owned(),
            raw_value: Vec::new(),
        };
        assert_eq!(
            header_value_string(&hv),
            "fallback",
            "should use value when raw_value is empty"
        );
    }

    #[test]
    fn header_value_string_non_utf8_raw_value() {
        let hv = HeaderValue {
            key: "x-bin".to_owned(),
            value: String::new(),
            raw_value: vec![0xFF, 0xFE],
        };
        let s = header_value_string(&hv);
        assert!(
            s.contains('\u{FFFD}'),
            "non-UTF-8 bytes should produce replacement chars, got: {s}"
        );
    }

    // -------------------------------------------------------------------------
    // is_pseudo_header
    // -------------------------------------------------------------------------

    #[test]
    fn pseudo_header_detected() {
        assert!(is_pseudo_header(":method"), ":method is a pseudo-header");
        assert!(is_pseudo_header(":path"), ":path is a pseudo-header");
    }

    #[test]
    fn regular_header_not_pseudo() {
        assert!(!is_pseudo_header("host"), "host is not a pseudo-header");
        assert!(!is_pseudo_header("x-custom"), "x-custom is not a pseudo-header");
    }

    // -------------------------------------------------------------------------
    // resolve_append_action
    // -------------------------------------------------------------------------

    #[test]
    fn resolve_append_action_default_is_append() {
        let hvo = HeaderValueOption {
            header: None,
            append: None,
            append_action: 0,
        };
        let action = resolve_append_action(&hvo);
        assert_eq!(
            action,
            HeaderAppendAction::AppendIfExistsOrAdd,
            "default should be AppendIfExistsOrAdd"
        );
    }

    #[test]
    fn resolve_append_action_explicit_overwrite() {
        let hvo = HeaderValueOption {
            header: None,
            append: None,
            append_action: HeaderAppendAction::OverwriteIfExistsOrAdd as i32,
        };
        let action = resolve_append_action(&hvo);
        assert_eq!(
            action,
            HeaderAppendAction::OverwriteIfExistsOrAdd,
            "explicit overwrite should be OverwriteIfExistsOrAdd"
        );
    }

    #[test]
    fn resolve_append_action_deprecated_append_false() {
        let hvo = HeaderValueOption {
            header: None,
            append: Some(false),
            append_action: 0,
        };
        let action = resolve_append_action(&hvo);
        assert_eq!(
            action,
            HeaderAppendAction::OverwriteIfExistsOrAdd,
            "deprecated append=false should map to OverwriteIfExistsOrAdd"
        );
    }

    // -------------------------------------------------------------------------
    // immediate_to_rejection
    // -------------------------------------------------------------------------

    #[test]
    fn immediate_to_rejection_default_status_200() {
        let imm = ImmediateResponse {
            status: None,
            headers: None,
            body: String::new(),
            grpc_status: None,
            details: String::new(),
        };
        let action = immediate_to_rejection(&imm);
        assert!(
            matches!(&action, FilterAction::Reject(r) if r.status == 200),
            "missing status should default to 200"
        );
    }

    #[test]
    fn immediate_to_rejection_custom_status() {
        let imm = ImmediateResponse {
            status: Some(HttpStatus { code: 403 }),
            headers: None,
            body: String::new(),
            grpc_status: None,
            details: String::new(),
        };
        let action = immediate_to_rejection(&imm);
        assert!(
            matches!(&action, FilterAction::Reject(r) if r.status == 403),
            "should use the configured status code"
        );
    }

    #[test]
    fn immediate_to_rejection_negative_status_falls_back_to_500() {
        let imm = ImmediateResponse {
            status: Some(HttpStatus { code: -1 }),
            headers: None,
            body: String::new(),
            grpc_status: None,
            details: String::new(),
        };
        let action = immediate_to_rejection(&imm);
        assert!(
            matches!(&action, FilterAction::Reject(r) if r.status == 500),
            "negative status should fall back to 500"
        );
    }

    #[test]
    fn immediate_to_rejection_with_body() {
        let imm = ImmediateResponse {
            status: Some(HttpStatus { code: 429 }),
            headers: None,
            body: "rate limited".to_owned(),
            grpc_status: None,
            details: String::new(),
        };
        let action = immediate_to_rejection(&imm);
        assert!(
            matches!(&action, FilterAction::Reject(r) if r.body.as_deref() == Some(b"rate limited".as_slice())),
            "rejection should include the body"
        );
    }
}
