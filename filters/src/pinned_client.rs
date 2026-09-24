// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Pinned `reqwest` client for cloud credential endpoints.
//!
//! Only the experimental `azure_ad` and `gcp_adc` filters need a `reqwest`
//! client, so it is built here behind their feature gates rather than in
//! `praxis-ai-apis`, keeping `reqwest` out of builds that do not enable them.

use std::net::IpAddr;

use praxis_ai_apis::callout_target::{AddressPolicy, validate_configured_http_target, validate_resolved_addrs};
use praxis_filter::FilterError;

/// Build a redirect-free, proxy-free `reqwest` client pinned to the address
/// set that passed the shared connect-time policy.
///
/// This adapter exists for protocol clients that cannot use
/// [`SubRequestClient`](praxis_core::subrequest::SubRequestClient), notably
/// cloud credential endpoints.
///
/// # Errors
///
/// Returns [`FilterError`] for invalid targets, failed or timed-out DNS,
/// disallowed resolved addresses, or client construction failure.
#[expect(
    clippy::too_many_lines,
    reason = "validation, one-time resolution, pinning, and client hardening are one security boundary"
)]
pub(crate) async fn build_pinned_reqwest_client(
    filter_name: &str,
    target: &str,
    policy: AddressPolicy,
    timeout: std::time::Duration,
) -> Result<reqwest::Client, FilterError> {
    let started = std::time::Instant::now();
    let parsed = validate_configured_http_target(filter_name, target, policy)?;
    let host = parsed
        .host_str()
        .ok_or_else(|| -> FilterError { format!("{filter_name}: target URL must include a host").into() })?;
    // `Url::host_str()` serializes IPv6 hosts with brackets; remove them
    // before passing the host to `IpAddr` parsing or DNS resolution.
    let host_without_brackets = host
        .strip_prefix('[')
        .and_then(|host| host.strip_suffix(']'))
        .unwrap_or(host);
    let port = parsed
        .port_or_known_default()
        .ok_or_else(|| -> FilterError { format!("{filter_name}: target URL has no usable port").into() })?;

    let mut builder = reqwest::Client::builder()
        .no_proxy()
        .redirect(reqwest::redirect::Policy::none());

    if host_without_brackets.parse::<IpAddr>().is_err() {
        let resolved = tokio::time::timeout(timeout, tokio::net::lookup_host((host_without_brackets, port)))
            .await
            .map_err(|_elapsed| -> FilterError { format!("{filter_name}: DNS resolution timed out").into() })?
            .map_err(|error| -> FilterError {
                format!("{filter_name}: DNS resolution failed for {host}: {error}").into()
            })?
            .collect::<Vec<_>>();
        let validated = validate_resolved_addrs(filter_name, &resolved, policy)?;
        builder = builder.resolve_to_addrs(host, &validated);
    }

    let remaining = timeout.checked_sub(started.elapsed()).ok_or_else(|| -> FilterError {
        format!("{filter_name}: callout deadline exceeded during target resolution").into()
    })?;
    builder
        .timeout(remaining)
        .build()
        .map_err(|error| format!("{filter_name}: failed to build HTTP client: {error}").into())
}
