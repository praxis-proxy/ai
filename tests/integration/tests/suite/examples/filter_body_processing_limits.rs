// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Per-filter processing limits remain independent of shared pipeline buffering.

use std::collections::HashMap;

use praxis_core::config::Config;
use praxis_test_utils::{
    StatefulCapturingBackend, free_port, http_send, load_example_config, parse_status, start_proxy,
};

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// Shipped examples and the request models that exercise their transformations.
const CASES: [(&str, &str, &str); 3] = [
    ("prompt-enrichment.yaml", "prompt_enrich", "gpt-4o"),
    ("model-to-provider.yaml", "model_to_provider", "claude-sonnet-4-5"),
    (
        "llmisvc-model-provider-resolver.yaml",
        "llmisvc_model_provider_resolver",
        "publishers/rhoai/models/granite-3.1-8b",
    ),
];

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[test]
fn filter_body_processing_limits_inherit_pipeline_above_ten_mib() {
    for (example, filter, model) in CASES {
        let backend = StatefulCapturingBackend::new(vec![(200, "ok".to_owned())]).start_with_shutdown();
        let config = config(example, filter, backend.port(), None);
        let proxy = start_proxy(&config);
        let raw = http_send(proxy.addr(), &request(&model_body(model, 12_582_912), false));
        assert_eq!(
            parse_status(&raw),
            200,
            "{filter}: body above 10 MiB must be processed: {raw}"
        );
        let requests = backend.requests();
        assert_eq!(requests.len(), 1, "{filter}: forward exactly once");
        let body: serde_json::Value = serde_json::from_str(&requests[0].body).unwrap();
        assert!(body["padding"].as_str().unwrap().len() > 10_485_760);
        assert_transformed(filter, &body);
    }
}

#[test]
fn filter_body_processing_limits_enforce_explicit_boundary_with_larger_shared_buffer() {
    for (example, filter, model) in CASES {
        for chunked in [false, true] {
            let backend = StatefulCapturingBackend::new(vec![(200, "ok".to_owned())]).start_with_shutdown();
            let config = config(example, filter, backend.port(), Some(1024));
            let proxy = start_proxy(&config);
            let raw = http_send(proxy.addr(), &request(&model_body(model, 1024), chunked));
            assert_eq!(parse_status(&raw), 200, "{filter}: exact boundary succeeds: {raw}");
            let raw = http_send(proxy.addr(), &request(&model_body(model, 1025), chunked));
            assert_eq!(parse_status(&raw), 413, "{filter}: one byte over rejects: {raw}");
            let requests = backend.requests();
            assert_eq!(requests.len(), 1, "{filter}: rejected body must not reach upstream");
            assert_transformed(filter, &serde_json::from_str(&requests[0].body).unwrap());
        }
    }
}

#[test]
fn filter_body_processing_limits_preserve_pipeline_request_and_buffer_bounds() {
    for (example, filter, model) in CASES {
        for (request_cap, buffer_cap) in [(1024, 2048), (2048, 1024)] {
            for chunked in [false, true] {
                let backend = StatefulCapturingBackend::new(vec![(200, "ok".to_owned())]).start_with_shutdown();
                let mut config = config(example, filter, backend.port(), None);
                config.body_limits.max_request_bytes = Some(request_cap);
                config.body_limits.max_buffer_bytes = buffer_cap.try_into().unwrap();
                let proxy = start_proxy(&config);
                let raw = http_send(proxy.addr(), &request(&model_body(model, 512), chunked));
                assert_eq!(
                    parse_status(&raw),
                    200,
                    "{filter}: transformed body fits within pipeline bounds: {raw}"
                );
                let raw = http_send(proxy.addr(), &request(&model_body(model, 1025), chunked));
                assert_eq!(
                    parse_status(&raw),
                    413,
                    "{filter}: effective pipeline bound exceeded: {raw}"
                );
                assert_eq!(
                    backend.requests().len(),
                    1,
                    "{filter}: rejected body must not reach upstream"
                );
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Test Utilities
// -----------------------------------------------------------------------------

/// Load a shipped example and give the shared buffer much more room than the filter limit.
fn config(example: &str, filter: &str, backend: u16, limit: Option<usize>) -> Config {
    let config = load_example_config(
        example,
        free_port(),
        HashMap::from([
            ("127.0.0.1:3000", backend),
            ("127.0.0.1:9001", backend),
            ("127.0.0.1:9002", backend),
        ]),
    );
    let mut value = serde_yaml::to_value(config).unwrap();
    let filters = value["filter_chains"][0]["filters"].as_sequence_mut().unwrap();
    if let Some(limit) = limit {
        let target = filters
            .iter_mut()
            .find(|entry| entry["filter"].as_str() == Some(filter))
            .unwrap();
        target["max_body_bytes"] = serde_yaml::to_value(limit).unwrap();
    }
    // The resolver example already includes this inspector. In the others, put it
    // after the transforming filter so enrichment still runs before model inspection.
    if limit.is_some()
        && !filters
            .iter()
            .any(|entry| entry["filter"].as_str() == Some("model_to_header"))
    {
        filters.insert(1, serde_yaml::from_str("filter: model_to_header").unwrap());
    }
    let mut config: Config = serde_yaml::from_value(value).unwrap();
    config.body_limits.max_request_bytes = Some(33_554_432);
    config.body_limits.max_buffer_bytes = 33_554_432.try_into().unwrap();
    config
}

/// Assert the existing behavior still happens after successful size validation.
fn assert_transformed(filter: &str, body: &serde_json::Value) {
    match filter {
        "prompt_enrich" => {
            assert_eq!(body["model"], "gpt-4o", "prompt enrichment must preserve the model");
            assert_eq!(
                body["messages"][0]["role"], "system",
                "prompt enrichment must prepend its system message"
            );
            assert_eq!(
                body["messages"][1]["content"], "Remember to cite your sources.",
                "prompt enrichment must append its configured reminder"
            );
        },
        "model_to_provider" => assert_eq!(
            body["model"], "claude-sonnet-4-5@20250929",
            "model_to_provider must rewrite to the configured provider model"
        ),
        "llmisvc_model_provider_resolver" => assert_eq!(
            body["model"], "granite-3.1-8b",
            "the resolver must rewrite the publisher ID to the short model name"
        ),
        _ => unreachable!("known test case"),
    }
}

/// Valid chat JSON with the model after padding to require complete inspection.
fn model_body(model: &str, size: usize) -> String {
    let prefix = r#"{"padding":""#;
    let suffix = format!(r#"","model":"{model}","messages":[]}}"#);
    format!("{prefix}{}{suffix}", "x".repeat(size - prefix.len() - suffix.len()))
}

/// Exercise both known and unknown transfer lengths.
fn request(body: &str, chunked: bool) -> String {
    let head =
        "POST /v1/messages HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nConnection: close\r\n";
    if chunked {
        let (first, second) = body.split_at(body.len() / 2);
        format!(
            "{head}Transfer-Encoding: chunked\r\n\r\n{:x}\r\n{first}\r\n{:x}\r\n{second}\r\n0\r\n\r\n",
            first.len(),
            second.len()
        )
    } else {
        format!("{head}Content-Length: {}\r\n\r\n{body}", body.len())
    }
}
