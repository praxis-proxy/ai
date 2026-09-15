// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Effective request headers for body-phase outbound callouts.

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
pub fn effective_callout_headers<'a>(
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
