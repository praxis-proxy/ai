// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Effective request headers for outbound callouts.

use std::borrow::Cow;

use http::{HeaderMap, HeaderValue};
use praxis_filter::{HttpFilterContext, TrustedHeaderMutation};

/// Apply trusted body pre-read mutations to the header view used by a callout.
///
/// `StreamBuffer` body filters run before the normal request-header phase. The
/// protocol applies their ordered mutation log only after pre-read completes,
/// so a later body filter must use this overlay instead of reading the original
/// request map directly. Otherwise it could forward a client-supplied value
/// that an earlier security filter removed or replaced.
///
/// The input remains borrowed when no mutation exists. When mutations are
/// present, the header map is copied once at this outbound ownership boundary
/// and then updated in the same order used by the protocol layer.
#[must_use]
pub fn effective_body_callout_headers<'a>(
    ctx: &HttpFilterContext<'_>,
    mut headers: Cow<'a, HeaderMap>,
) -> Cow<'a, HeaderMap> {
    let mutations = ctx.prior_pre_read_mutations.iter().chain(ctx.pre_read_mutations.iter());

    for mutation in mutations {
        let headers = headers.to_mut();
        match mutation {
            TrustedHeaderMutation::Remove(name) => {
                headers.remove(name);
            },
            TrustedHeaderMutation::Set(name, value) => {
                headers.insert(name.clone(), value.clone());
            },
            TrustedHeaderMutation::Add(name, value) => match HeaderValue::from_str(value) {
                Ok(value) => {
                    headers.append(name.clone(), value);
                },
                Err(error) => {
                    tracing::warn!(
                        header = %name,
                        error = %error,
                        "skipping invalid trusted pre-read add mutation for callout"
                    );
                },
            },
        }
    }

    headers
}

/// Apply pending request-phase mutations to the header view used by a callout.
///
/// Request filters queue removals, replacements, and promoted headers for the
/// protocol layer to commit after the pipeline finishes. A callout in that
/// same pipeline must overlay those queues to see the request that will be
/// sent upstream, including changes made by earlier security filters.
///
/// The input remains borrowed when no mutation exists. When mutations are
/// present, the header map is copied once at this outbound ownership boundary.
/// The remove, set, then extra order matches the protocol layer. Extra headers
/// use replacement semantics for the same reason they do there.
#[must_use]
pub fn effective_request_callout_headers<'a>(
    ctx: &HttpFilterContext<'_>,
    mut headers: Cow<'a, HeaderMap>,
) -> Cow<'a, HeaderMap> {
    for name in &ctx.request_headers_to_remove {
        headers.to_mut().remove(name);
    }
    for (name, value) in &ctx.request_headers_to_set {
        headers.to_mut().insert(name.clone(), value.clone());
    }
    for (name, value) in &ctx.extra_request_headers {
        match (
            http::HeaderName::from_bytes(name.as_bytes()),
            HeaderValue::from_str(value),
        ) {
            (Ok(name), Ok(value)) => {
                headers.to_mut().insert(name, value);
            },
            (name_result, value_result) => {
                tracing::warn!(
                    header = %name,
                    name_err = ?name_result.err(),
                    value_err = ?value_result.err(),
                    "skipping invalid promoted header mutation for callout"
                );
            },
        }
    }

    headers
}
