// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Unit tests for the model catalog filter.

#![expect(clippy::expect_used, clippy::panic, clippy::indexing_slicing, reason = "tests")]

use std::{collections::BTreeSet, sync::Arc, time::Duration};

use bytes::Bytes;
use http::Method;
use praxis_filter::{FilterAction, HttpFilter as _, TerminalResponse};
use serde_json::{Value, json};
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{header, method, path},
};

use super::{
    CatalogRoute, CatalogSource, ModelCatalogFilter,
    config::{DefaultVisibility, ModelCatalogConfig, validate_config},
    discovery::Discovery,
    listing::Catalog,
    visibility::{PrincipalView, Visibility},
};
use crate::test_utils::{make_filter_context, make_request};

const CATALOG: &str = r#"
created: 1700000000
models:
  - id: gpt-4o
    owned_by: openai
    provider: openai
    version: "2024-08-06"
    capabilities: [chat_completions, responses]
    cost_tier: standard
    aliases: [gpt4o]
  - id: meta-llama/Llama-3.1-8B
    created: 1720000000
  - id: claude-sonnet-4-5
    provider: vertex
    capabilities: [messages]
    visible_to:
      subjects: [alice]
      roles: [ml-eng]
      teams: [research]
"#;

fn parse(yaml: &str) -> ModelCatalogConfig {
    serde_yaml::from_str(yaml).expect("valid catalog yaml")
}

fn build(yaml: &str) -> ModelCatalogFilter {
    let cfg = parse(yaml);
    validate_config(&cfg).expect("valid catalog config");
    ModelCatalogFilter::build(cfg, crate::isolated_subrequest_client(1)).expect("catalog builds")
}

impl ModelCatalogFilter {
    fn static_catalog(&self) -> &Catalog {
        match &self.source {
            CatalogSource::Static(catalog) => catalog,
            CatalogSource::Discovered(_) => panic!("expected a static catalog"),
        }
    }

    fn list_body(&self, principal: Option<PrincipalView<'_>>) -> Bytes {
        self.static_catalog().list_body(&self.overlay, principal)
    }

    fn discovery(&self) -> &Arc<Discovery> {
        match &self.source {
            CatalogSource::Discovered(discovery) => discovery,
            CatalogSource::Static(_) => panic!("expected a discovered catalog"),
        }
    }

    /// Refresh on every request instead of after the configured interval.
    fn refresh_always(mut self) -> Self {
        if let CatalogSource::Discovered(discovery) = &mut self.source {
            Arc::get_mut(discovery)
                .expect("discovery is not shared yet")
                .set_refresh_interval(Duration::ZERO);
        }
        self
    }
}

fn validation_error(yaml: &str) -> String {
    validate_config(&parse(yaml))
        .expect_err("config should be rejected")
        .to_string()
}

fn body_json(response: &TerminalResponse) -> Value {
    serde_json::from_slice(response.body.as_ref().expect("response body")).expect("json body")
}

fn listed_ids(response: &TerminalResponse) -> Vec<String> {
    let body = body_json(response);
    assert_eq!(body["object"], "list");
    body["data"]
        .as_array()
        .expect("data array")
        .iter()
        .map(|model| model["id"].as_str().expect("id").to_owned())
        .collect()
}

fn principal<'a>(subject: &'a str, roles: &'a BTreeSet<String>, teams: &'a BTreeSet<String>) -> PrincipalView<'a> {
    PrincipalView { subject, roles, teams }
}

fn set(values: &[&str]) -> BTreeSet<String> {
    values.iter().map(|&value| value.to_owned()).collect()
}

async fn respond(filter: &ModelCatalogFilter, method: Method, path: &str) -> FilterAction {
    let request = make_request(method, path);
    let mut ctx = make_filter_context(&request);
    filter.on_request(&mut ctx).await.expect("on_request")
}

fn terminal(action: FilterAction) -> TerminalResponse {
    match action {
        FilterAction::TerminalResponse(response) => *response,
        _ => panic!("expected a terminal response"),
    }
}

// -----------------------------------------------------------------------------
// Routing
// -----------------------------------------------------------------------------

#[test]
fn route_matches_list_and_retrieve_paths() {
    let filter = build(CATALOG);
    assert_eq!(filter.route("/v1/models"), Some(CatalogRoute::List));
    assert_eq!(filter.route("/v1/models/"), Some(CatalogRoute::List));
    assert_eq!(
        filter.route("/v1/models/gpt-4o"),
        Some(CatalogRoute::Retrieve("gpt-4o"))
    );
    assert_eq!(
        filter.route("/v1/models/meta-llama/Llama-3.1-8B"),
        Some(CatalogRoute::Retrieve("meta-llama/Llama-3.1-8B"))
    );
    assert_eq!(filter.route("/v1/modelsx"), None);
    assert_eq!(filter.route("/v1/chat/completions"), None);
}

#[test]
fn route_honours_custom_list_path() {
    let filter = build("list_path: /catalog/models\nmodels: [{id: m}]");
    assert_eq!(filter.route("/catalog/models"), Some(CatalogRoute::List));
    assert_eq!(filter.route("/v1/models"), None);
}

#[tokio::test]
async fn non_catalog_paths_continue() {
    let filter = build(CATALOG);
    assert!(matches!(
        respond(&filter, Method::POST, "/v1/chat/completions").await,
        FilterAction::Continue
    ));
}

#[tokio::test]
async fn non_get_methods_are_rejected_with_405() {
    let filter = build(CATALOG);
    let response = terminal(respond(&filter, Method::POST, "/v1/models").await);
    assert_eq!(response.status, 405);
    assert_eq!(response.headers["allow"], "GET");
    assert_eq!(body_json(&response)["error"]["code"], "method_not_allowed");
}

// -----------------------------------------------------------------------------
// Listing
// -----------------------------------------------------------------------------

#[tokio::test]
async fn anonymous_list_shows_only_public_models() {
    let filter = build(CATALOG);
    let response = terminal(respond(&filter, Method::GET, "/v1/models").await);
    assert_eq!(response.status, 200);
    assert_eq!(response.headers["content-type"], "application/json");
    assert_eq!(response.headers["cache-control"], "private, no-store");
    assert_eq!(listed_ids(&response), ["gpt-4o", "gpt4o", "meta-llama/Llama-3.1-8B"]);
}

#[test]
fn list_includes_restricted_models_for_matching_principal() {
    let filter = build(CATALOG);
    let empty = set(&[]);
    let body = filter.list_body(Some(principal("alice", &empty, &empty)));
    let response = TerminalResponse::new(200).with_body(body);
    assert_eq!(
        listed_ids(&response),
        ["gpt-4o", "gpt4o", "meta-llama/Llama-3.1-8B", "claude-sonnet-4-5"]
    );
}

#[test]
fn list_serializes_openai_shape_and_catalog_metadata() {
    let filter = build(CATALOG);
    let response = TerminalResponse::new(200).with_body(filter.list_body(None));
    let body = body_json(&response);

    let gpt = &body["data"][0];
    assert_eq!(gpt["object"], "model");
    assert_eq!(gpt["created"], 1_700_000_000);
    assert_eq!(gpt["owned_by"], "openai");
    assert_eq!(gpt["praxis"]["provider"], "openai");
    assert_eq!(gpt["praxis"]["version"], "2024-08-06");
    assert_eq!(
        gpt["praxis"]["capabilities"],
        serde_json::json!(["chat_completions", "responses"])
    );
    assert_eq!(gpt["praxis"]["cost_tier"], "standard");
    assert_eq!(gpt["praxis"]["aliases"], serde_json::json!(["gpt4o"]));

    let alias = &body["data"][1];
    assert_eq!(alias["praxis"]["alias_of"], "gpt-4o");
    assert!(alias["praxis"].get("aliases").is_none(), "aliases do not list aliases");

    let llama = &body["data"][2];
    assert_eq!(llama["created"], 1_720_000_000);
    assert_eq!(llama["owned_by"], "praxis");
    assert!(
        llama.get("praxis").is_none(),
        "entries without metadata omit the praxis key"
    );
}

#[test]
fn owned_by_defaults_to_provider() {
    let filter = build(CATALOG);
    let empty = set(&[]);
    let response = TerminalResponse::new(200).with_body(filter.list_body(Some(principal("alice", &empty, &empty))));
    assert_eq!(body_json(&response)["data"][3]["owned_by"], "vertex");
}

// -----------------------------------------------------------------------------
// Retrieval
// -----------------------------------------------------------------------------

#[tokio::test]
async fn retrieve_returns_visible_model_and_decodes_path() {
    let filter = build(CATALOG);
    let response = terminal(respond(&filter, Method::GET, "/v1/models/gpt4o").await);
    assert_eq!(response.status, 200);
    assert_eq!(body_json(&response)["id"], "gpt4o");

    let encoded = terminal(respond(&filter, Method::GET, "/v1/models/meta-llama%2FLlama-3.1-8B").await);
    assert_eq!(encoded.status, 200);
    assert_eq!(body_json(&encoded)["id"], "meta-llama/Llama-3.1-8B");
}

#[tokio::test]
async fn hidden_and_unknown_models_return_identical_404() {
    let filter = build(CATALOG);
    let hidden = terminal(respond(&filter, Method::GET, "/v1/models/claude-sonnet-4-5").await);
    let unknown = terminal(respond(&filter, Method::GET, "/v1/models/no-such-model").await);
    assert_eq!(hidden.status, 404);
    assert_eq!(unknown.status, 404);

    let hidden = body_json(&hidden);
    let unknown = body_json(&unknown);
    assert_eq!(hidden["error"]["code"], "model_not_found");
    assert_eq!(hidden["error"]["type"], unknown["error"]["type"]);
    assert_eq!(
        hidden["error"]["message"],
        "The model 'claude-sonnet-4-5' does not exist"
    );
}

#[test]
fn retrieve_restricted_model_for_matching_principal() {
    let filter = build(CATALOG);
    let empty = set(&[]);
    let response = filter.retrieve(
        filter.static_catalog(),
        "claude-sonnet-4-5",
        Some(principal("alice", &empty, &empty)),
    );
    assert_eq!(response.status, 200);
}

// -----------------------------------------------------------------------------
// Visibility
// -----------------------------------------------------------------------------

#[test]
fn restricted_visibility_matches_subject_role_or_team() {
    let rule = Visibility::compile(
        parse("models: [{id: m, visible_to: {subjects: [alice], roles: [ml-eng], teams: [research]}}]")
            .models
            .pop()
            .and_then(|entry| entry.visible_to),
        DefaultVisibility::Public,
    );
    let empty = set(&[]);
    let role = set(&["ml-eng"]);
    let team = set(&["research"]);
    let other = set(&["sales"]);

    assert!(rule.allows(Some(principal("alice", &empty, &empty))));
    assert!(rule.allows(Some(principal("bob", &role, &empty))));
    assert!(rule.allows(Some(principal("bob", &empty, &team))));
    assert!(!rule.allows(Some(principal("bob", &other, &other))));
    assert!(!rule.allows(None));
}

#[test]
fn authenticated_default_visibility_requires_identity() {
    let filter = build("default_visibility: authenticated\nmodels: [{id: m}]");
    let empty = set(&[]);
    let anonymous = TerminalResponse::new(200).with_body(filter.list_body(None));
    let signed_in = TerminalResponse::new(200).with_body(filter.list_body(Some(principal("bob", &empty, &empty))));
    assert!(listed_ids(&anonymous).is_empty());
    assert_eq!(listed_ids(&signed_in), ["m"]);
}

// -----------------------------------------------------------------------------
// Configuration
// -----------------------------------------------------------------------------

#[test]
fn from_config_registers_valid_catalog() {
    let value: serde_yaml::Value = serde_yaml::from_str(CATALOG).expect("yaml");
    let filter = ModelCatalogFilter::from_config(&value).expect("filter builds");
    assert_eq!(filter.name(), "model_catalog");
}

#[test]
fn unknown_fields_are_rejected() {
    let value: serde_yaml::Value = serde_yaml::from_str("models: [{id: m, colour: blue}]").expect("yaml");
    assert!(ModelCatalogFilter::from_config(&value).is_err());
}

#[test]
fn unknown_capability_is_rejected() {
    let value: serde_yaml::Value = serde_yaml::from_str("models: [{id: m, capabilities: [telepathy]}]").expect("yaml");
    assert!(ModelCatalogFilter::from_config(&value).is_err());
}

#[test]
fn validation_rejects_invalid_catalogs() {
    for (yaml, expected) in [
        (
            "models: []",
            "models must not be empty when no upstreams are configured",
        ),
        (
            "models: [{id: m, upstream_id: n}]",
            "upstream_id for 'm' requires upstreams",
        ),
        ("list_path: v1/models\nmodels: [{id: m}]", "must start with '/'"),
        ("list_path: /v1/models/\nmodels: [{id: m}]", "must not end with '/'"),
        ("models: [{id: ''}]", "must not be empty"),
        ("models: [{id: 'a b'}]", "cannot appear in a path"),
        ("models: [{id: m}, {id: m}]", "duplicate model id 'm'"),
        ("models: [{id: m, aliases: [n]}, {id: n}]", "duplicate model id 'n'"),
        ("models: [{id: m, visible_to: {}}]", "must list at least one"),
        (
            "models: [{id: m, cost_tier: ''}]",
            "cost_tier for 'm' must not be empty",
        ),
    ] {
        let error = validation_error(yaml);
        assert!(error.contains(expected), "{yaml}: expected '{expected}' in '{error}'");
    }
}

#[test]
#[expect(clippy::too_many_lines, reason = "table of rejected upstream configs")]
fn validation_rejects_invalid_upstreams() {
    for (yaml, expected) in [
        (
            "upstreams: [{name: a, url: 'http://127.0.0.1/v1/models'}]\nrefresh_interval_secs: 0",
            "refresh_interval_secs must be at least 1",
        ),
        (
            "upstreams: [{name: a, url: 'http://127.0.0.1/v1/models'}]\nunconfigured_models: hide",
            "requires at least one models entry",
        ),
        (
            "upstreams: [{name: a, url: 'http://h/v1/models'}, {name: a, url: 'http://h/v1/models'}]",
            "duplicate upstream name 'a'",
        ),
        (
            "upstreams: [{name: '', url: 'http://h/v1/models'}]",
            "names must not be empty",
        ),
        (
            "upstreams: [{name: a, url: 'http://h/v1/models'}]\nmodels: [{id: m, upstream_id: x}, {id: n, upstream_id: x}]",
            "upstream id 'x' is claimed by more than one entry",
        ),
        (
            "upstreams: [{name: a, url: 'http://h/v1/models', headers: [{name: x-key}]}]",
            "needs exactly one of value or value_env_var",
        ),
        (
            "upstreams: [{name: a, url: 'http://h/v1/models', headers: [{name: x-key, value: v, value_env_var: V}]}]",
            "needs exactly one of value or value_env_var",
        ),
    ] {
        let error = validation_error(yaml);
        assert!(error.contains(expected), "{yaml}: expected '{expected}' in '{error}'");
    }
}

// -----------------------------------------------------------------------------
// Upstream discovery
// -----------------------------------------------------------------------------

/// Serve `models` from `GET /v1/models` on a fresh mock upstream.
async fn upstream(models: Value) -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"object": "list", "data": models})))
        .mount(&server)
        .await;
    server
}

/// Upstream entry pointing at a mock server's model list.
fn upstream_yaml(name: &str, server: &MockServer, extra: &str) -> String {
    format!(
        "  - name: {name}\n    url: {}/v1/models\n    allow_private: true\n{extra}",
        server.uri()
    )
}

fn build_discovered(upstreams: &[String], rest: &str) -> ModelCatalogFilter {
    build(&format!("upstreams:\n{}{rest}", upstreams.concat()))
}

async fn list_ids(filter: &ModelCatalogFilter) -> Vec<String> {
    let response = terminal(respond(filter, Method::GET, "/v1/models").await);
    assert_eq!(response.status, 200, "list should succeed");
    listed_ids(&response)
}

/// OpenAI and vLLM upstreams overlaid by gpt-4o (with an alias), a restricted
/// llama entry mapped by `upstream_id`, and an entry no upstream reports. The
/// servers are returned so they outlive the filter's fetches.
async fn overlay_fixture() -> (MockServer, MockServer, ModelCatalogFilter) {
    let openai = upstream(json!([
        {"id": "gpt-4o", "object": "model", "created": 1_715_000_000, "owned_by": "system"},
        {"id": "gpt-4o-mini", "object": "model", "created": 1_721_000_000, "owned_by": "system"},
    ]))
    .await;
    let vllm = upstream(json!([
        {"id": "meta-llama/Llama-3.1-70B-Instruct", "object": "model", "owned_by": "vllm"},
        {"id": "gpt-4o", "object": "model", "owned_by": "vllm"},
    ]))
    .await;
    let filter = build_discovered(
        &[
            upstream_yaml("openai", &openai, "    provider: openai\n"),
            upstream_yaml("vllm", &vllm, "    provider: vllm\n"),
        ],
        "
models:
  - id: gpt-4o
    capabilities: [chat_completions]
    aliases: [gpt4o]
  - id: llama-70b
    upstream_id: meta-llama/Llama-3.1-70B-Instruct
    visible_to: {subjects: [alice]}
  - id: retired-model
",
    );
    (openai, vllm, filter)
}

/// The discovered list body as `alice` sees it.
async fn alice_list(filter: &ModelCatalogFilter) -> TerminalResponse {
    let snapshot = filter.discovery().snapshot().await.expect("snapshot");
    let catalog = snapshot.catalog.as_deref().expect("catalog");
    let empty = set(&[]);
    TerminalResponse::new(200).with_body(catalog.list_body(&filter.overlay, Some(principal("alice", &empty, &empty))))
}

#[tokio::test]
async fn discovered_catalog_lists_upstream_models_per_principal() {
    let (_openai, _vllm, filter) = overlay_fixture().await;
    assert_eq!(
        list_ids(&filter).await,
        ["gpt-4o", "gpt4o", "gpt-4o-mini"],
        "restricted entries are hidden, unreported entries are dropped, and the first upstream wins duplicates"
    );
    assert_eq!(
        listed_ids(&alice_list(&filter).await),
        ["gpt-4o", "gpt4o", "gpt-4o-mini", "llama-70b"],
        "an entry's upstream_id is listed under its client-facing id"
    );
}

#[tokio::test]
async fn discovered_models_merge_upstream_fields_with_entry_metadata() {
    let (_openai, _vllm, filter) = overlay_fixture().await;
    let body = body_json(&alice_list(&filter).await);

    let gpt = &body["data"][0];
    assert_eq!(gpt["created"], 1_715_000_000, "created falls back to the upstream");
    assert_eq!(gpt["owned_by"], "system", "owned_by falls back to the upstream");
    assert_eq!(gpt["praxis"]["capabilities"], json!(["chat_completions"]));

    let mini = &body["data"][2];
    assert_eq!(
        mini["praxis"]["provider"], "openai",
        "unconfigured models take the upstream provider"
    );
    assert!(
        mini["praxis"].get("capabilities").is_none(),
        "no capability is claimed for unconfigured models"
    );
    assert_eq!(body["data"][3]["owned_by"], "vllm");
}

#[tokio::test]
async fn claimed_upstream_ids_and_unreported_entries_are_not_listed() {
    let (_openai, _vllm, filter) = overlay_fixture().await;
    let raw = terminal(respond(&filter, Method::GET, "/v1/models/meta-llama%2FLlama-3.1-70B-Instruct").await);
    assert_eq!(raw.status, 404, "upstream ids claimed by an entry are not listed");
    let retired = terminal(respond(&filter, Method::GET, "/v1/models/retired-model").await);
    assert_eq!(retired.status, 404, "entries no upstream reports are not listed");
}

#[tokio::test]
async fn hidden_unconfigured_models_are_not_listed() {
    let server = upstream(json!([{"id": "a"}, {"id": "b"}])).await;
    let filter = build_discovered(
        &[upstream_yaml("u", &server, "")],
        "unconfigured_models: hide\nmodels: [{id: b}]\n",
    );
    assert_eq!(list_ids(&filter).await, ["b"]);
}

#[tokio::test]
async fn upstream_credentials_are_sent() {
    let server = MockServer::start().await;
    // cargo sets CARGO_PKG_NAME for test processes, so no unsafe set_var.
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .and(header("authorization", "Bearer praxis-ai-filters"))
        .and(header("anthropic-version", "2023-06-01"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"data": [{"id": "m"}]})))
        .mount(&server)
        .await;
    let filter = build_discovered(
        &[upstream_yaml(
            "u",
            &server,
            "    bearer_token_env_var: CARGO_PKG_NAME\n    headers: [{name: anthropic-version, value: '2023-06-01'}]\n",
        )],
        "",
    );
    assert_eq!(list_ids(&filter).await, ["m"]);
}

#[test]
fn missing_upstream_credential_is_rejected_at_build() {
    let yaml = "upstreams: [{name: u, url: 'http://127.0.0.1/v1/models', allow_private: true, \
                bearer_token_env_var: DEFINITELY_NOT_SET_MODEL_CATALOG_XYZ123}]";
    let value: serde_yaml::Value = serde_yaml::from_str(yaml).expect("yaml");
    let error = ModelCatalogFilter::from_config(&value)
        .err()
        .expect("missing env var")
        .to_string();
    assert!(error.contains("is not set"), "{error}");
}

#[test]
fn private_upstream_requires_opt_in() {
    let value: serde_yaml::Value =
        serde_yaml::from_str("upstreams: [{name: u, url: 'http://127.0.0.1/v1/models'}]").expect("yaml");
    assert!(ModelCatalogFilter::from_config(&value).is_err());
}

#[tokio::test]
async fn catalog_is_unavailable_until_an_upstream_answers() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .respond_with(ResponseTemplate::new(500))
        .mount(&server)
        .await;
    let filter = build_discovered(&[upstream_yaml("u", &server, "")], "");

    for path in ["/v1/models", "/v1/models/m"] {
        let response = terminal(respond(&filter, Method::GET, path).await);
        assert_eq!(response.status, 503, "{path}");
        let body = body_json(&response);
        assert_eq!(body["error"]["code"], "model_catalog_unavailable");
        assert_eq!(body["error"]["type"], "server_error");
    }
}

#[tokio::test]
async fn failed_refresh_keeps_last_known_list_and_removals_apply() {
    let server = MockServer::start().await;
    for response in [
        ResponseTemplate::new(200).set_body_json(json!({"data": [{"id": "a"}, {"id": "b"}]})),
        ResponseTemplate::new(503),
        ResponseTemplate::new(200).set_body_json(json!({"data": [{"id": "a"}]})),
    ] {
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .respond_with(response)
            .up_to_n_times(1)
            .mount(&server)
            .await;
    }
    let filter = build_discovered(&[upstream_yaml("u", &server, "")], "").refresh_always();
    let discovery = filter.discovery();

    discovery.refresh_if_due().await;
    assert_eq!(snapshot_ids(&filter), ["a", "b"]);

    discovery.refresh_if_due().await;
    assert_eq!(
        snapshot_ids(&filter),
        ["a", "b"],
        "a failed refresh keeps the last known list"
    );

    discovery.refresh_if_due().await;
    assert_eq!(
        snapshot_ids(&filter),
        ["a"],
        "models the upstream stops reporting are dropped"
    );
}

#[tokio::test]
async fn stale_snapshot_is_served_while_refreshing_in_background() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"data": [{"id": "old"}]})))
        .up_to_n_times(1)
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"data": [{"id": "new"}]})))
        .mount(&server)
        .await;
    let filter = build_discovered(&[upstream_yaml("u", &server, "")], "").refresh_always();

    assert_eq!(list_ids(&filter).await, ["old"], "the first request fetches inline");
    assert_eq!(
        list_ids(&filter).await,
        ["old"],
        "a due snapshot is still served while the refresh runs"
    );

    for _ in 0..200 {
        if snapshot_ids(&filter) == ["new"] {
            return;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("the background refresh never replaced the snapshot");
}

/// IDs in the current discovered snapshot, without triggering a refresh.
fn snapshot_ids(filter: &ModelCatalogFilter) -> Vec<String> {
    let snapshot = filter.discovery().current().expect("snapshot");
    let catalog = snapshot.catalog.as_deref().expect("catalog");
    listed_ids(&TerminalResponse::new(200).with_body(catalog.list_body(&filter.overlay, None)))
}
