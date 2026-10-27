// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Unit tests for the model catalog filter.

#![expect(clippy::expect_used, clippy::panic, clippy::indexing_slicing, reason = "tests")]

use std::collections::BTreeSet;

use http::Method;
use praxis_filter::{FilterAction, HttpFilter as _, TerminalResponse};
use serde_json::Value;

use super::{
    CatalogRoute, ModelCatalogFilter,
    config::{DefaultVisibility, ModelCatalogConfig, validate_config},
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
    ModelCatalogFilter::build(cfg).expect("catalog builds")
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
    let response = filter.retrieve("claude-sonnet-4-5", Some(principal("alice", &empty, &empty)));
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
        ("models: []", "models must not be empty"),
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
