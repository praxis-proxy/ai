// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Fire-and-forget delivery of metering `CloudEvents`.

use std::time::Duration;

use bytes::Bytes;
use metrics::counter;
use praxis_ai_apis::{
    callout_target::AddressPolicy,
    subrequest::{self, SubRequest, SubRequestClient, SubRequestError, SubResponse},
};
use tracing::{trace, warn};

use super::METRIC_REPORT_FAILURES;
#[cfg(feature = "metering-kafka")]
use super::kafka::KafkaUsageEventSink;

/// Upper bound on metering service response bodies (HTTP sink only).
const MAX_CALLOUT_RESPONSE_BYTES: usize = 64 * 1024;

/// Where usage/error `CloudEvents` are sent after a request completes.
pub(super) enum UsageEventSink {
    /// Historical default: HTTP POST to `{metering_url}/api/v1/events`.
    Http(HttpUsageEventSink),
    /// Optional Kafka producer (requires the `metering-kafka` feature).
    #[cfg(feature = "metering-kafka")]
    Kafka(KafkaUsageEventSink),
}

/// HTTP POST transport for metering `CloudEvents`.
pub(super) struct HttpUsageEventSink {
    /// Connect-time policy for the metering events URL.
    address_policy: AddressPolicy,
    /// Shared HTTP client used for the fire-and-forget POST.
    client: SubRequestClient,
    /// Absolute URL of `{metering_url}/api/v1/events`.
    events_url: String,
    /// Per-request timeout for the events POST.
    timeout: Duration,
}

impl HttpUsageEventSink {
    /// Build a sink that POSTs to `{metering_url}/api/v1/events`.
    pub(super) fn new(
        client: SubRequestClient,
        metering_url: &str,
        timeout: Duration,
        address_policy: AddressPolicy,
    ) -> Self {
        let events_url = format!("{}/api/v1/events", metering_url.trim_end_matches('/'));
        Self {
            address_policy,
            client,
            events_url,
            timeout,
        }
    }

    /// Spawn a fire-and-forget HTTP POST of `body`.
    pub(super) fn deliver(&self, body: Vec<u8>, mut headers: http::HeaderMap) {
        headers.insert(
            http::header::CONTENT_TYPE,
            http::HeaderValue::from_static("application/json"),
        );

        let client = self.client.clone();
        let url = self.events_url.clone();
        let timeout = self.timeout;
        let address_policy = self.address_policy;

        tokio::spawn(async move {
            let request = SubRequest {
                method: http::Method::POST,
                uri: http::Uri::default(),
                headers,
                body: Bytes::from(body),
            };

            report_http_delivery(
                client
                    .execute_url(
                        &url,
                        request,
                        subrequest::address_policy_hook(address_policy),
                        subrequest::UrlResolutionPolicy::OperatorCached,
                        timeout,
                        MAX_CALLOUT_RESPONSE_BYTES,
                        None,
                    )
                    .await
                    .map_err(subrequest::map_url_error),
            );
        });
    }
}

/// Log the outcome of an HTTP usage report and count failures.
fn report_http_delivery(result: Result<SubResponse, SubRequestError>) {
    match result {
        Ok(resp) if is_success(resp.status) => {
            trace!(status = resp.status, "metering usage report sent");
        },
        Ok(resp) => {
            counter!(METRIC_REPORT_FAILURES).increment(1);
            warn!(status = resp.status, "metering usage report rejected");
        },
        Err(error) => {
            counter!(METRIC_REPORT_FAILURES).increment(1);
            warn!(%error, "metering usage report failed");
        },
    }
}

/// Whether an HTTP status code is in the 2xx range.
fn is_success(status: u16) -> bool {
    (200..300).contains(&status)
}
