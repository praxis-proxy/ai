// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! `cargo xtask visualize-config <path> --output <path>` renders an arbitrary
//! Praxis configuration as a single, self-contained, offline HTML document.
//!
//! Unlike [`crate::flow_visualizer`], which merges a curated sidecar to produce
//! the rich full-flow visualizer, this command has no sidecar: it shows only the
//! mechanically inferred topology (parsed through the pinned `praxis_core` via
//! [`crate::flow_graph::FlowGraph`]) plus, for each filter, a one-line
//! description **iff** this repository defines that filter (see
//! [`crate::filter_docs::filter_descriptions`]). Filters with no known
//! definition are labelled "semantics unavailable" and their raw configuration
//! is shown verbatim — redacted for sensitive-looking values. The command never
//! fabricates a behavioral explanation for a structure it cannot ground.
//!
//! The extracted "document model" can also be written as JSON (`--json`) and
//! dropped back into any generated HTML: the browser re-renders from that model
//! without ever parsing YAML, so there is no second, drift-prone parser.

use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

use clap::Parser;
use serde_json::{Map, Value as Json};

use crate::{
    filter_docs,
    flow_graph::{FlowGraph, FlowNode},
    html,
};

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// The generic, self-contained HTML shell with a single `@@MODEL@@` placeholder.
const TEMPLATE: &str = include_str!("assets/generic_visualizer.html");

/// The placeholder the template exposes for the embedded document model.
const MODEL_PLACEHOLDER: &str = "@@MODEL@@";

/// The sentinel substituted for redacted configuration values. Must match the
/// `REDACTED` constant the template's renderer highlights.
const REDACTED: &str = "«redacted»";

/// Case-insensitive substrings that mark a configuration key as sensitive; its
/// value is replaced with [`REDACTED`] before the model is embedded.
const SENSITIVE_KEY_FRAGMENTS: &[&str] = &[
    "access_key",
    "apikey",
    "api_key",
    "authorization",
    "bearer",
    "client_secret",
    "cookie",
    "credential",
    "passwd",
    "password",
    "private_key",
    "secret",
    "session_key",
    "token",
];

// -----------------------------------------------------------------------------
// CLI
// -----------------------------------------------------------------------------

/// CLI arguments for `cargo xtask visualize-config`.
#[derive(Parser)]
pub(crate) struct Args {
    /// Path to the Praxis config to visualize.
    config: PathBuf,
    /// Path to write the generated self-contained HTML document.
    #[arg(long)]
    output: PathBuf,
    /// Also write the extracted document-model JSON here, for offline reuse via
    /// the generated page's drag-and-drop / file picker.
    #[arg(long)]
    json: Option<PathBuf>,
}

// -----------------------------------------------------------------------------
// Entry point
// -----------------------------------------------------------------------------

/// Render the requested config, exiting with status 1 on any failure.
pub(crate) fn run(args: &Args) {
    if let Err(err) = generate(args) {
        eprintln!("visualize-config: {err}");
        std::process::exit(1);
    }
}

/// Parse the config, build the document model, and write the HTML (and JSON).
fn generate(args: &Args) -> Result<(), String> {
    let graph = FlowGraph::from_file(&args.config)?;
    let descriptions = filter_docs::filter_descriptions(&workspace_root());
    let source = args.config.display().to_string();
    let model = build_model(&source, &graph, &descriptions);

    if let Some(json_path) = &args.json {
        let pretty = serde_json::to_string_pretty(&model).map_err(|err| err.to_string())?;
        write_file(json_path, &pretty)?;
        println!("wrote {}", json_path.display());
    }

    let html = embed(TEMPLATE, &model)?;
    write_file(&args.output, &html)?;
    println!("wrote {}", args.output.display());
    Ok(())
}

// -----------------------------------------------------------------------------
// Model construction
// -----------------------------------------------------------------------------

/// Build the complete document model rendered by the template.
fn build_model(source: &str, graph: &FlowGraph, descriptions: &BTreeMap<String, String>) -> Json {
    let chains = graph
        .filter_chains
        .iter()
        .map(|chain| build_chain(graph, &chain.name, descriptions))
        .collect();
    let listeners = serde_json::to_value(&graph.listeners).unwrap_or(Json::Null);
    let clusters = redact(&serde_json::to_value(&graph.clusters).unwrap_or(Json::Null));
    obj(vec![
        ("source", Json::String(source.to_owned())),
        (
            "provenance",
            Json::String("extracted by cargo xtask visualize-config via praxis_core".to_owned()),
        ),
        ("listeners", listeners),
        ("clusters", clusters),
        ("insecure_options", redact(&graph.insecure_options)),
        ("runtime", redact(&graph.runtime)),
        ("chains", Json::Array(chains)),
    ])
}

/// Build one flattened chain, hoisting IRR inference-step filters inline.
fn build_chain(graph: &FlowGraph, chain_name: &str, descriptions: &BTreeMap<String, String>) -> Json {
    let nodes = graph.flatten_chain(chain_name).unwrap_or_default();
    let filters = nodes.iter().map(|node| build_filter_node(node, descriptions)).collect();
    obj(vec![
        ("name", Json::String(chain_name.to_owned())),
        ("filters", Json::Array(filters)),
    ])
}

/// Build one filter node, attaching a description only for known filters.
fn build_filter_node(node: &FlowNode, descriptions: &BTreeMap<String, String>) -> Json {
    let known = descriptions.contains_key(&node.filter_type);
    let description = descriptions.get(&node.filter_type).filter(|text| !text.is_empty());
    obj(vec![
        ("order", Json::from(node.order)),
        ("depth", Json::from(node.depth)),
        ("filter_type", Json::String(node.filter_type.clone())),
        ("name", opt_str(node.name.as_deref())),
        ("irr_step", opt_str(node.irr_step.as_deref())),
        ("known", Json::Bool(known)),
        ("description", opt_str(description.map(String::as_str))),
        ("conditions", redact(&node.conditions)),
        ("branch_chains", redact(&node.branch_chains)),
        ("config", redact(&node.config)),
    ])
}

// -----------------------------------------------------------------------------
// Redaction
// -----------------------------------------------------------------------------

/// Deep-copy a JSON value, replacing sensitive-keyed values with [`REDACTED`].
fn redact(value: &Json) -> Json {
    match value {
        Json::Object(map) => Json::Object(redact_map(map)),
        Json::Array(items) => Json::Array(items.iter().map(redact).collect()),
        other => other.clone(),
    }
}

/// Redact one object: sensitive keys collapse to the sentinel (unless the value
/// is a plain config knob), others recurse, then the `{name, value}` header shape
/// is handled as a special case.
fn redact_map(map: &Map<String, Json>) -> Map<String, Json> {
    let mut out = Map::new();
    for (key, value) in map {
        let redacted = if is_sensitive_key(key) {
            redact_sensitive_value(value)
        } else {
            redact(value)
        };
        out.insert(key.clone(), redacted);
    }
    redact_named_value(&mut out);
    out
}

/// Collapse a sensitive-keyed value to the sentinel, but keep scalar knobs.
///
/// A key can contain a sensitive fragment (`token`) yet name an ordinary numeric
/// or boolean limit — `max_tokens: 500`, `reserved_tokens: 8`, `refresh_token:
/// true`. Those carry no secret, so preserve them; only strings, arrays, and
/// objects can hold actual credential material, so collapse those.
fn redact_sensitive_value(value: &Json) -> Json {
    match value {
        Json::Number(_) | Json::Bool(_) | Json::Null => value.clone(),
        _ => Json::String(REDACTED.to_owned()),
    }
}

/// Redact a `{ name: "<sensitive>", value: "<secret>" }` pairing.
///
/// Header-style config hides the secret in a `value` whose sibling `name`
/// identifies it (e.g. `name: Authorization`, `value: "Bearer …"`). The per-key
/// pass misses it because neither `name` nor `value` is itself a sensitive key.
/// Only a string `value` is collapsed, so a `{name, value: 30}` numeric pair is
/// left intact.
fn redact_named_value(out: &mut Map<String, Json>) {
    let named_secret = out.get("name").and_then(Json::as_str).is_some_and(is_sensitive_key);
    if named_secret
        && let Some(value) = out.get_mut("value")
        && value.is_string()
    {
        *value = Json::String(REDACTED.to_owned());
    }
}

/// Return `true` when a config key looks like it carries a secret. Hyphens are
/// normalized to underscores so `api-key` and `x-api-key` match the same
/// fragments as `api_key`.
fn is_sensitive_key(key: &str) -> bool {
    let lower = key.to_ascii_lowercase().replace('-', "_");
    SENSITIVE_KEY_FRAGMENTS.iter().any(|fragment| lower.contains(fragment))
}

// -----------------------------------------------------------------------------
// Embedding
// -----------------------------------------------------------------------------

/// Substitute the pretty-printed, HTML-safe model into the template.
///
/// The template must contain the placeholder exactly once, so a stray mention
/// (e.g. in a comment) can never divert the substitution to the wrong spot.
fn embed(template: &str, model: &Json) -> Result<String, String> {
    let count = template.matches(MODEL_PLACEHOLDER).count();
    if count != 1 {
        return Err(format!(
            "template must contain exactly one {MODEL_PLACEHOLDER} placeholder, found {count}"
        ));
    }
    let pretty = serde_json::to_string_pretty(model).map_err(|err| err.to_string())?;
    let safe = html::escape_json_for_script(&pretty);
    Ok(template.replacen(MODEL_PLACEHOLDER, &safe, 1))
}

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

/// Assemble a JSON object from ordered `(key, value)` entries.
fn obj(entries: Vec<(&str, Json)>) -> Json {
    let mut map = Map::new();
    for (key, value) in entries {
        map.insert(key.to_owned(), value);
    }
    Json::Object(map)
}

/// Render an optional string as a JSON string or `null`.
fn opt_str(value: Option<&str>) -> Json {
    value.map_or(Json::Null, |text| Json::String(text.to_owned()))
}

/// Write `contents` to `path`, mapping IO errors to an actionable message.
fn write_file(path: &Path, contents: &str) -> Result<(), String> {
    std::fs::write(path, contents).map_err(|err| format!("write {}: {err}", path.display()))
}

/// The workspace root (the parent of the `xtask` crate directory).
fn workspace_root() -> PathBuf {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set — run via `cargo xtask`");
    Path::new(&manifest_dir)
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .to_owned()
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::unwrap_used, clippy::expect_used, reason = "tests")]
mod tests {
    use super::*;

    /// The committed full-flow config, relative to the xtask crate root.
    const FULL_FLOW: &str = "../examples/configs/openai/responses/full-flow-agentic.yaml";

    /// The chain the full-flow visualizer flattens.
    const PIPELINE_CHAIN: &str = "full-flow-agentic-pipeline";

    /// A minimal valid config with the given filters in one chain.
    fn minimal_config(filters: &[&str]) -> String {
        let mut body = String::from(
            "listeners:\n  - name: l\n    address: 127.0.0.1:8080\n    filter_chains: [c]\nfilter_chains:\n  - name: c\n    filters:\n",
        );
        for filter in filters {
            body.push_str("      - filter: ");
            body.push_str(filter);
            body.push('\n');
        }
        body
    }

    /// Locate a filter node in a built model by its `filter_type`.
    fn find_filter<'a>(model: &'a Json, chain: &str, filter_type: &str) -> Option<&'a Json> {
        model
            .get("chains")?
            .as_array()?
            .iter()
            .find(|c| c.get("name").and_then(Json::as_str) == Some(chain))?
            .get("filters")?
            .as_array()?
            .iter()
            .find(|f| f.get("filter_type").and_then(Json::as_str) == Some(filter_type))
    }

    #[test]
    fn full_flow_model_marks_known_and_unknown_filters() {
        let graph = FlowGraph::from_file(Path::new(FULL_FLOW)).expect("config parses");
        let descriptions = filter_docs::filter_descriptions(&workspace_root());
        let model = build_model("full-flow", &graph, &descriptions);

        let known = find_filter(&model, PIPELINE_CHAIN, "openai_responses_proxy").expect("proxy present");
        assert_eq!(
            known.get("known").and_then(Json::as_bool),
            Some(true),
            "AI-repo filter is known"
        );
        assert!(
            known
                .get("description")
                .and_then(Json::as_str)
                .is_some_and(|d| !d.is_empty()),
            "known filter carries a description"
        );

        let unknown = find_filter(&model, PIPELINE_CHAIN, "trace_context").expect("trace_context present");
        assert_eq!(
            unknown.get("known").and_then(Json::as_bool),
            Some(false),
            "core filter is unknown"
        );
        assert_eq!(
            unknown.get("description"),
            Some(&Json::Null),
            "unknown filter has no fabricated prose"
        );
    }

    #[test]
    fn unknown_filter_renders_structure_without_fabrication() {
        let graph = FlowGraph::from_yaml_str(&minimal_config(&["trace_context"])).expect("valid config");
        let empty = BTreeMap::new();
        let model = build_model("min", &graph, &empty);
        let node = find_filter(&model, "c", "trace_context").expect("filter present");
        assert_eq!(node.get("known").and_then(Json::as_bool), Some(false));
        assert_eq!(node.get("description"), Some(&Json::Null));
        assert!(node.get("config").is_some(), "raw config is still surfaced");
    }

    #[test]
    fn redaction_hides_sensitive_values_and_keeps_the_rest() {
        let input = serde_json::json!({
            "authorization": "Bearer sk-live-123",
            "nested": { "api_key": "secret", "timeout_ms": 5000 },
            "endpoints": ["10.0.0.1:80"],
        });
        let out = redact(&input);
        assert_eq!(out.get("authorization"), Some(&Json::String(REDACTED.to_owned())));
        assert_eq!(out.pointer("/nested/api_key"), Some(&Json::String(REDACTED.to_owned())));
        assert_eq!(out.pointer("/nested/timeout_ms").and_then(Json::as_u64), Some(5000));
        assert_eq!(out.pointer("/endpoints/0").and_then(Json::as_str), Some("10.0.0.1:80"));
    }

    #[test]
    fn sensitive_key_detection() {
        for key in [
            "token",
            "API_KEY",
            "api-key",
            "x-api-key",
            "Authorization",
            "client_secret",
            "db_password",
        ] {
            assert!(is_sensitive_key(key), "{key} should be sensitive");
        }
        for key in ["address", "timeout_ms", "max_iterations", "endpoints"] {
            assert!(!is_sensitive_key(key), "{key} should not be sensitive");
        }
    }

    #[test]
    fn sensitive_keyed_numeric_knobs_are_preserved() {
        // A key can carry a sensitive fragment yet name a plain scalar limit; the
        // knob is not a secret and must survive, while string material does not.
        let input = serde_json::json!({
            "max_tokens": 500,
            "reserved_tokens": 8,
            "refresh_token": true,
            "access_token": "sk-live-123",
        });
        let out = redact(&input);
        assert_eq!(
            out.get("max_tokens").and_then(Json::as_u64),
            Some(500),
            "numeric knob kept"
        );
        assert_eq!(
            out.get("reserved_tokens").and_then(Json::as_u64),
            Some(8),
            "numeric knob kept"
        );
        assert_eq!(
            out.get("refresh_token").and_then(Json::as_bool),
            Some(true),
            "boolean knob kept"
        );
        assert_eq!(
            out.get("access_token"),
            Some(&Json::String(REDACTED.to_owned())),
            "string secret redacted"
        );
    }

    #[test]
    fn named_value_secret_pairs_are_redacted() {
        // Header-style config: the secret hides in `value`, identified only by a
        // sibling `name`. A string value is redacted; a numeric one is a knob.
        let input = serde_json::json!({
            "headers": [
                { "name": "Authorization", "value": "Bearer sk-live-123" },
                { "name": "X-Api-Key", "value": "secret-key" },
                { "name": "X-Timeout", "value": 30 },
                { "name": "Accept", "value": "application/json" },
            ]
        });
        let out = redact(&input);
        assert_eq!(
            out.pointer("/headers/0/value"),
            Some(&Json::String(REDACTED.to_owned())),
            "authorization value redacted"
        );
        assert_eq!(
            out.pointer("/headers/1/value"),
            Some(&Json::String(REDACTED.to_owned())),
            "api-key value redacted"
        );
        assert_eq!(
            out.pointer("/headers/2/value").and_then(Json::as_u64),
            Some(30),
            "numeric value under a sensitive name is a knob, not a secret"
        );
        assert_eq!(
            out.pointer("/headers/3/value").and_then(Json::as_str),
            Some("application/json"),
            "non-sensitive name leaves its value intact"
        );
    }

    #[test]
    fn embed_substitutes_the_placeholder() {
        let graph = FlowGraph::from_yaml_str(&minimal_config(&["trace_context"])).expect("valid config");
        let model = build_model("min", &graph, &BTreeMap::new());
        let html = embed(TEMPLATE, &model).expect("embeds");
        assert!(!html.contains(MODEL_PLACEHOLDER), "placeholder is consumed");
        assert!(html.contains("const MODEL ="), "model is injected into the script");
    }

    #[test]
    fn model_construction_is_deterministic() {
        let graph = FlowGraph::from_yaml_str(&minimal_config(&["trace_context", "state_owner"])).expect("valid config");
        let descriptions = BTreeMap::new();
        let first = build_model("min", &graph, &descriptions);
        let second = build_model("min", &graph, &descriptions);
        assert_eq!(first, second, "model construction must be deterministic");
    }

    #[test]
    fn embedded_model_round_trips_through_html_escaping() {
        // A filter config value carrying the exact characters that could
        // prematurely close the <script> element or open an HTML comment.
        let yaml = "listeners:\n  - name: l\n    address: 127.0.0.1:8080\n    filter_chains: [c]\nfilter_chains:\n  - name: c\n    filters:\n      - filter: trace_context\n        note: \"</script><!-- a & b -->\"\n";
        let graph = FlowGraph::from_yaml_str(yaml).expect("valid config");
        let model = build_model("safe", &graph, &BTreeMap::new());
        let html = embed(TEMPLATE, &model).expect("embeds");

        // The injected model neither introduces a second closing tag nor leaves
        // the placeholder behind: the literal "</script>" was neutralized.
        assert_eq!(
            html.matches("</script>").count(),
            1,
            "only the template's own closing tag survives"
        );
        assert!(!html.contains(MODEL_PLACEHOLDER), "placeholder consumed");

        // Recover the embedded model text and prove it round-trips: reversing
        // the HTML escaping yields JSON that parses back to exactly the model.
        let marker = "const MODEL = ";
        let start = html.find(marker).expect("model assignment present") + marker.len();
        let tail = html.get(start..).expect("model tail");
        let end = tail.find(";\n").expect("assignment terminator");
        let embedded = tail.get(..end).expect("model text");
        let unescaped = embedded
            .replace("\\u003c", "<")
            .replace("\\u003e", ">")
            .replace("\\u0026", "&");
        let parsed: Json = serde_json::from_str(&unescaped).expect("embedded model is valid JSON");
        assert_eq!(parsed, model, "the embedded model round-trips losslessly");
    }

    #[test]
    fn arbitrary_config_with_branches_surfaces_branch_topology() {
        let yaml = "listeners:\n  - name: l\n    address: 127.0.0.1:8080\n    filter_chains: [c]\nfilter_chains:\n  - name: c\n    filters:\n      - filter: headers\n        branch_chains:\n          - name: bypass\n            rejoin: terminal\n            chains:\n              - name: bypass-chain\n                filters:\n                  - filter: trace_context\n      - filter: state_owner\n";
        let graph = FlowGraph::from_yaml_str(yaml).expect("branched config parses");
        let model = build_model("branched", &graph, &BTreeMap::new());

        // The branch topology is surfaced on the carrier node...
        let carrier = find_filter(&model, "c", "headers").expect("carrier present");
        let branches = carrier
            .get("branch_chains")
            .and_then(Json::as_array)
            .expect("branch_chains array");
        assert_eq!(branches.len(), 1, "the branch chain is surfaced on the carrier node");

        // ...but the branch's inner filter is captured as data, not hoisted:
        // the chain still lists exactly the two declared top-level filters.
        let chain = model
            .get("chains")
            .and_then(Json::as_array)
            .expect("chains array")
            .iter()
            .find(|c| c.get("name").and_then(Json::as_str) == Some("c"))
            .expect("chain c present");
        let filters = chain.get("filters").and_then(Json::as_array).expect("filters array");
        assert_eq!(filters.len(), 2, "branch inner filters are not hoisted into the chain");
    }
}
