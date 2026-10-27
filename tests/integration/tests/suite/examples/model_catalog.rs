// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Integration tests for the `model-catalog` example configuration.
//!
//! The catalog answers `GET /v1/models` from config, filtered by the subject
//! `basic_auth` verified, and must never contact an upstream to do so. Listed
//! IDs must stay routable through the `model_to_provider` chain behind it.

use std::collections::HashMap;

use praxis_test_utils::{
    ProxyGuard, StatefulCapturingBackend, StatefulCapturingGuard, basic_auth_header, example_config_path, free_port,
    http_send, json_post_with_header, load_example_config, parse_body, parse_header, parse_status, start_proxy,
};
use serde_json::Value;

const BACKEND_RESPONSE: &str = r#"{"id":"chatcmpl-1","object":"chat.completion","choices":[]}"#;

/// Running example proxy plus the backends behind both clusters.
struct Harness {
    proxy: ProxyGuard,
    openai: StatefulCapturingGuard,
    vertex: StatefulCapturingGuard,
    password: String,
}

impl Harness {
    fn start() -> Self {
        let openai = StatefulCapturingBackend::new(vec![(200, BACKEND_RESPONSE.to_owned())]).start_with_shutdown();
        let vertex = StatefulCapturingBackend::new(vec![(200, BACKEND_RESPONSE.to_owned())]).start_with_shutdown();
        let config = load_example_config(
            "model-catalog.yaml",
            free_port(),
            HashMap::from([("127.0.0.1:3000", openai.port()), ("127.0.0.1:3001", vertex.port())]),
        );
        Self {
            proxy: start_proxy(&config),
            openai,
            vertex,
            password: password_from_example(),
        }
    }

    fn get(&self, path: &str, username: &str) -> String {
        let authorization = basic_auth_header(username, &self.password);
        http_send(
            self.proxy.addr(),
            &format!(
                "GET {path} HTTP/1.1\r\n\
                 Host: localhost\r\n\
                 Authorization: {authorization}\r\n\
                 Connection: close\r\n\r\n"
            ),
        )
    }

    fn listed_ids(&self, username: &str) -> Vec<String> {
        let raw = self.get("/v1/models", username);
        assert_eq!(parse_status(&raw), 200, "the catalog list should succeed");
        let body: Value = serde_json::from_str(&parse_body(&raw)).expect("catalog list is JSON");
        body["data"]
            .as_array()
            .expect("catalog list has a data array")
            .iter()
            .map(|model| model["id"].as_str().expect("model id").to_owned())
            .collect()
    }

    fn assert_no_upstream_contact(&self) {
        assert!(
            self.openai.requests().is_empty() && self.vertex.requests().is_empty(),
            "catalog requests must be answered by the proxy"
        );
    }
}

/// Read the shared example password so it is not duplicated as a literal.
fn password_from_example() -> String {
    let path = example_config_path("model-catalog.yaml");
    let yaml = std::fs::read_to_string(&path).unwrap_or_else(|error| panic!("read {path}: {error}"));
    yaml.lines()
        .find_map(|line| {
            line.trim()
                .strip_prefix("password:")
                .map(|value| value.trim().trim_matches('"').to_owned())
        })
        .expect("model-catalog.yaml has a password field")
}

#[test]
fn example_config_model_catalog_lists_models_per_principal() {
    let harness = Harness::start();

    assert_eq!(harness.listed_ids("viewer"), ["gpt-4o", "gpt4o"]);
    assert_eq!(
        harness.listed_ids("ml-engineer"),
        ["gpt-4o", "gpt4o", "claude-sonnet-4-5"]
    );

    let raw = harness.get("/v1/models", "viewer");
    assert_eq!(
        parse_header(&raw, "cache-control").as_deref(),
        Some("private, no-store"),
        "principal-specific listings must not be shared by caches"
    );
    harness.assert_no_upstream_contact();
}

#[test]
fn example_config_model_catalog_hides_restricted_model_lookup() {
    let harness = Harness::start();

    let hidden = harness.get("/v1/models/claude-sonnet-4-5", "viewer");
    let unknown = harness.get("/v1/models/no-such-model", "viewer");
    assert_eq!(parse_status(&hidden), 404, "hidden models look unknown");
    assert_eq!(parse_status(&unknown), 404);
    let hidden: Value = serde_json::from_str(&parse_body(&hidden)).expect("404 is JSON");
    assert_eq!(hidden["error"]["code"], "model_not_found");

    let visible = harness.get("/v1/models/claude-sonnet-4-5", "ml-engineer");
    assert_eq!(parse_status(&visible), 200);
    let visible: Value = serde_json::from_str(&parse_body(&visible)).expect("model is JSON");
    assert_eq!(visible["id"], "claude-sonnet-4-5");
    assert_eq!(visible["owned_by"], "anthropic");
    assert_eq!(visible["praxis"]["capabilities"], serde_json::json!(["messages"]));

    harness.assert_no_upstream_contact();
}

#[test]
fn example_config_model_catalog_requires_authentication() {
    let harness = Harness::start();
    let raw = http_send(
        harness.proxy.addr(),
        "GET /v1/models HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n",
    );
    assert_eq!(parse_status(&raw), 401, "basic_auth runs before the catalog");
    harness.assert_no_upstream_contact();
}

#[test]
fn example_config_model_catalog_alias_routes_like_listed_id() {
    let harness = Harness::start();
    assert!(harness.listed_ids("viewer").iter().any(|id| id == "gpt4o"));

    let authorization = format!("Authorization: {}", basic_auth_header("viewer", &harness.password));
    let raw = http_send(
        harness.proxy.addr(),
        &json_post_with_header(
            "/v1/chat/completions",
            r#"{"model":"gpt4o","messages":[{"role":"user","content":"hi"}]}"#,
            &authorization,
        ),
    );
    assert_eq!(parse_status(&raw), 200, "a listed alias should route");

    let forwarded = harness
        .openai
        .requests()
        .into_iter()
        .find(|request| request.uri.starts_with("/v1/chat/completions"))
        .expect("the openai cluster receives the aliased request");
    let body: Value = serde_json::from_str(&forwarded.body).expect("forwarded body is JSON");
    assert_eq!(body["model"], "gpt-4o", "model_to_provider resolves the alias");
    assert!(harness.vertex.requests().is_empty());
}
