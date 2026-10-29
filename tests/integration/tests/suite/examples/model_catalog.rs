// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Integration tests for the `model-catalog` example configuration.
//!
//! The catalog answers `GET /v1/models` from the model lists the upstreams
//! report, overlaid with configured metadata and filtered by the subject
//! `basic_auth` verified. Catalog requests are never forwarded; the only
//! upstream traffic they cause is the catalog's own model list fetch. Listed
//! IDs must stay routable through the `model_to_provider` chain behind it.

use std::collections::HashMap;

use praxis_test_utils::{
    ProxyGuard, StatefulCapturingBackend, StatefulCapturingGuard, basic_auth_header, example_config_path, free_port,
    http_send, json_post_with_header, load_example_config, parse_body, parse_header, parse_status, start_proxy,
};
use serde_json::Value;

const BACKEND_RESPONSE: &str = r#"{"id":"chatcmpl-1","object":"chat.completion","choices":[]}"#;

/// Model list the `openai` upstream reports. `gpt-4o-mini` has no catalog
/// entry and is listed as-is.
const OPENAI_MODELS: &str = r#"{"object":"list","data":[
    {"id":"gpt-4o","object":"model","created":1715367049,"owned_by":"system"},
    {"id":"gpt-4o-mini","object":"model","created":1721172741,"owned_by":"system"}]}"#;

/// Model list the `vllm` upstream reports, under its own model ID.
const VLLM_MODELS: &str = r#"{"object":"list","data":[
    {"id":"meta-llama/Llama-3.1-70B-Instruct","object":"model","created":1735000000,"owned_by":"vllm"}]}"#;

/// Running example proxy plus the backends behind both clusters.
struct Harness {
    proxy: ProxyGuard,
    openai: StatefulCapturingGuard,
    vllm: StatefulCapturingGuard,
    password: String,
}

impl Harness {
    /// Each backend answers the catalog's model list fetch first, then one
    /// inference request.
    fn start() -> Self {
        let openai = StatefulCapturingBackend::new(vec![
            (200, OPENAI_MODELS.to_owned()),
            (200, BACKEND_RESPONSE.to_owned()),
        ])
        .start_with_shutdown();
        let vllm =
            StatefulCapturingBackend::new(vec![(200, VLLM_MODELS.to_owned()), (200, BACKEND_RESPONSE.to_owned())])
                .start_with_shutdown();
        let config = load_example_config(
            "model-catalog.yaml",
            free_port(),
            HashMap::from([("127.0.0.1:3000", openai.port()), ("127.0.0.1:3001", vllm.port())]),
        );
        Self {
            proxy: start_proxy(&config),
            openai,
            vllm,
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

    /// Assert each upstream saw at most the catalog's single model list
    /// fetch, so catalog requests were answered by the proxy.
    fn assert_only_model_list_fetches(&self) {
        for (name, backend) in [("openai", &self.openai), ("vllm", &self.vllm)] {
            let requests = backend.requests();
            assert!(requests.len() <= 1, "{name} saw more than the model list fetch");
            for request in requests {
                assert_eq!(
                    (request.method.as_str(), request.uri.as_str()),
                    ("GET", "/v1/models"),
                    "{name} must only see the catalog's model list fetch"
                );
            }
        }
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

    assert_eq!(harness.listed_ids("viewer"), ["gpt-4o", "gpt4o", "gpt-4o-mini"]);
    assert_eq!(
        harness.listed_ids("ml-engineer"),
        ["gpt-4o", "gpt4o", "gpt-4o-mini", "llama-70b"]
    );

    let raw = harness.get("/v1/models", "viewer");
    assert_eq!(
        parse_header(&raw, "cache-control").as_deref(),
        Some("private, no-store"),
        "principal-specific listings must not be shared by caches"
    );
    harness.assert_only_model_list_fetches();
    assert_eq!(
        harness.openai.requests().len(),
        1,
        "the model list is fetched once and cached"
    );
}

#[test]
fn example_config_model_catalog_overlays_discovered_models() {
    let harness = Harness::start();

    let raw = harness.get("/v1/models/gpt-4o", "viewer");
    assert_eq!(parse_status(&raw), 200);
    let gpt: Value = serde_json::from_str(&parse_body(&raw)).expect("model is JSON");
    assert_eq!(gpt["created"], 1_715_367_049, "created comes from the upstream");
    assert_eq!(gpt["owned_by"], "system", "owned_by comes from the upstream");
    assert_eq!(gpt["praxis"]["cost_tier"], "standard", "metadata comes from the entry");

    let raw = harness.get("/v1/models/gpt-4o-mini", "viewer");
    assert_eq!(parse_status(&raw), 200, "unconfigured upstream models are listed");
    let mini: Value = serde_json::from_str(&parse_body(&raw)).expect("model is JSON");
    assert_eq!(mini["praxis"]["provider"], "openai");
    assert!(mini["praxis"].get("capabilities").is_none());

    let raw = harness.get("/v1/models/meta-llama%2FLlama-3.1-70B-Instruct", "ml-engineer");
    assert_eq!(
        parse_status(&raw),
        404,
        "an upstream ID claimed by an entry is not listed"
    );

    harness.assert_only_model_list_fetches();
}

#[test]
fn example_config_model_catalog_hides_restricted_model_lookup() {
    let harness = Harness::start();

    let hidden = harness.get("/v1/models/llama-70b", "viewer");
    let unknown = harness.get("/v1/models/no-such-model", "viewer");
    assert_eq!(parse_status(&hidden), 404, "hidden models look unknown");
    assert_eq!(parse_status(&unknown), 404);
    let hidden: Value = serde_json::from_str(&parse_body(&hidden)).expect("404 is JSON");
    assert_eq!(hidden["error"]["code"], "model_not_found");

    let visible = harness.get("/v1/models/llama-70b", "ml-engineer");
    assert_eq!(parse_status(&visible), 200);
    let visible: Value = serde_json::from_str(&parse_body(&visible)).expect("model is JSON");
    assert_eq!(visible["id"], "llama-70b");
    assert_eq!(visible["owned_by"], "vllm");
    assert_eq!(
        visible["praxis"]["capabilities"],
        serde_json::json!(["chat_completions"])
    );

    harness.assert_only_model_list_fetches();
}

#[test]
fn example_config_model_catalog_requires_authentication() {
    let harness = Harness::start();
    let raw = http_send(
        harness.proxy.addr(),
        "GET /v1/models HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n",
    );
    assert_eq!(parse_status(&raw), 401, "basic_auth runs before the catalog");
    assert!(
        harness.openai.requests().is_empty() && harness.vllm.requests().is_empty(),
        "an unauthenticated request must not trigger discovery"
    );
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
    assert!(
        harness
            .vllm
            .requests()
            .iter()
            .all(|request| request.uri == "/v1/models"),
        "the vllm cluster only sees the model list fetch"
    );
}
