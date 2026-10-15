// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Routing-decision metrics for `intelligent_route`, read from the
//! health/metrics listener after real requests through the proxy.
//!
//! Part of the praxis-proxy/ai#74 observability baseline: every decision the
//! filter makes is counted by outcome, and every fresh selection by cluster,
//! so operators can see routing behaviour without reading debug logs.

use std::collections::HashMap;

use praxis_test_utils::{
    free_port, http_get_retry, http_send, load_example_config, parse_status, start_backend_with_shutdown, start_proxy,
};

const INFERENCE_EXAMPLE: &str = "intelligent-route-inference.yaml";

/// Endpoints as written in `intelligent-route-inference.yaml`.
const GRANITE_ENDPOINT: &str = "127.0.0.1:8001";
const LLAMA_ENDPOINT: &str = "127.0.0.1:8002";
const ADMIN_ENDPOINT: &str = "127.0.0.1:9901";

/// Build a raw chat-completions POST carrying `body`.
fn chat_post(body: &str) -> String {
    format!(
        "POST /v1/chat/completions HTTP/1.1\r\n\
         Host: localhost\r\n\
         Content-Type: application/json\r\n\
         Content-Length: {}\r\n\
         Connection: close\r\n\r\n\
         {body}",
        body.len()
    )
}

/// Value of the sample `name` whose label set equals `labels` (order
/// independent; the exposition prints labels in insertion order).
fn metric_value(body: &str, name: &str, labels: &[(&str, &str)]) -> Option<u64> {
    let mut expected: Vec<String> = labels.iter().map(|(k, v)| format!("{k}=\"{v}\"")).collect();
    expected.sort_unstable();
    body.lines().find_map(|line| {
        let (sample, value) = line.rsplit_once(' ')?;
        let (sample_name, sample_labels) = match sample.split_once('{') {
            Some((n, rest)) => (n, rest.strip_suffix('}')?),
            None => (sample, ""),
        };
        if sample_name != name {
            return None;
        }
        let mut actual: Vec<String> = sample_labels
            .split(',')
            .filter(|l| !l.is_empty())
            .map(str::to_owned)
            .collect();
        actual.sort_unstable();
        (actual == expected).then(|| value.trim().parse::<f64>().ok().map(|v| v.round() as u64))?
    })
}

/// Scrape `/metrics` from the health/metrics listener.
///
/// Praxis 0.7.4 serves `/metrics` on `admin.metrics_address`, not on the
/// `/api` admin address the example configures.
fn scrape(metrics_address: &str) -> String {
    let (status, body) = http_get_retry(metrics_address, "/metrics", None);
    assert_eq!(
        status, 200,
        "/metrics must be served on the metrics listener; got {status}\n{body}"
    );
    body
}

/// One routed request and one unknown-model request are counted by outcome,
/// the routed one is attributed to its cluster, and the snapshot size is
/// published.
///
/// The Prometheus recorder is process-global and other suites in this binary
/// start their own proxies, so the test asserts on the delta between two
/// scrapes rather than on absolute counts.
#[test]
fn decisions_are_counted_on_the_metrics_listener() {
    let granite = start_backend_with_shutdown("granite-backend");
    let llama = start_backend_with_shutdown("llama-backend");
    let (proxy_port, admin_port) = (free_port(), free_port());
    let metrics_address = format!("127.0.0.1:{}", free_port());

    let mut config = load_example_config(
        INFERENCE_EXAMPLE,
        proxy_port,
        HashMap::from([
            (GRANITE_ENDPOINT, granite.port()),
            (LLAMA_ENDPOINT, llama.port()),
            (ADMIN_ENDPOINT, admin_port),
        ]),
    );
    config.admin.metrics_address = Some(metrics_address.clone());
    let proxy = start_proxy(&config);
    let before = scrape(&metrics_address);

    let routed = http_send(proxy.addr(), &chat_post(r#"{"model":"granite-3.3-8b","messages":[]}"#));
    assert_eq!(parse_status(&routed), 200, "known model must route: {routed}");
    let unknown = http_send(
        proxy.addr(),
        &chat_post(r#"{"model":"gpt-4o-not-configured","messages":[]}"#),
    );
    assert_eq!(parse_status(&unknown), 404, "unknown model must fail closed: {unknown}");

    let after = scrape(&metrics_address);
    let delta = |name: &str, labels: &[(&str, &str)]| {
        metric_value(&after, name, labels).unwrap_or(0) - metric_value(&before, name, labels).unwrap_or(0)
    };

    assert!(
        delta(
            "praxis_ai_intelligent_route_decisions_total",
            &[("kind", "inference_model"), ("outcome", "routed")],
        ) >= 1,
        "the routed request is counted:\n{after}"
    );
    assert!(
        delta(
            "praxis_ai_intelligent_route_decisions_total",
            &[("kind", "inference_model"), ("outcome", "no_candidate")],
        ) >= 1,
        "the unknown model is counted:\n{after}"
    );
    assert!(
        delta(
            "praxis_ai_intelligent_route_selected_total",
            &[
                ("cluster", "granite-local"),
                ("site", "site-a"),
                ("selection_mode", "deterministic"),
            ],
        ) >= 1,
        "the fresh selection is attributed to its cluster:\n{after}"
    );
    assert!(
        metric_value(&after, "praxis_ai_intelligent_route_snapshot_candidates", &[]).is_some_and(|n| n >= 1),
        "the serving snapshot size is published:\n{after}"
    );
}
