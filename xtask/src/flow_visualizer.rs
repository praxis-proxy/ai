// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! `cargo xtask sync-flow-visualizers` keeps the checked-in flow-visualizer
//! HTML in structural agreement with the config it documents.
//!
//! The command parses each configured visualizer's source YAML through the
//! pinned `praxis_core` parser (via [`crate::flow_graph::FlowGraph`]), flattens
//! the documented filter chain, and cross-checks the mechanically inferred
//! topology against the data embedded in the rendered HTML:
//!
//! * every filter is present, in order, and of the expected type;
//! * every `load_balancer` cluster (however deeply nested) is present with the same endpoints;
//! * every numeric/scalar limit the HTML surfaces matches its config source.
//!
//! Only mechanically inferable structure is checked here. Curated behavioral
//! narration (scenario prose, wire payloads, timelines) lives in the visualizer
//! and is validated by the generator introduced in a later increment; this
//! command is the structural drift gate wired into `make lint`.

use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
};

use clap::Parser;
use serde_json::Value as Json;

use crate::flow_graph::{FlowGraph, FlowNode};

// -----------------------------------------------------------------------------
// Registry
// -----------------------------------------------------------------------------

/// One checked-in flow visualizer and the config it documents.
struct Visualizer {
    /// Source config, relative to the workspace root.
    config: &'static str,
    /// Rendered HTML, relative to the workspace root.
    html: &'static str,
    /// The filter chain the HTML flattens and documents.
    chain: &'static str,
    /// Scalar limits the HTML surfaces and the config source of each.
    knobs: &'static [KnobCheck],
}

/// A limit surfaced in the HTML `knobs` map and the config value it must match.
struct KnobCheck {
    /// Key in the HTML `PIPELINE.knobs` object.
    knob: &'static str,
    /// Where the authoritative value lives in the parsed config.
    source: KnobSource,
    /// Config key to read within that source.
    key: &'static str,
}

/// The config location that owns a knob's authoritative value.
enum KnobSource {
    /// A filter's opaque config, located by its (unique) filter type.
    Filter(&'static str),
    /// The global `insecure_options` block.
    InsecureOptions,
}

/// Every checked-in flow visualizer validated by this command.
const VISUALIZERS: &[Visualizer] = &[Visualizer {
    config: "examples/configs/openai/responses/full-flow-agentic.yaml",
    html: "examples/configs/openai/responses/full-flow-agentic.visualizer.html",
    chain: "full-flow-agentic-pipeline",
    knobs: FULL_FLOW_KNOBS,
}];

/// The scalar limits the full-flow visualizer surfaces, mapped to their config
/// sources. `max_iterations_constraint` is curated prose (a human explanation of
/// the `>= max_infer_iters + 1` invariant) and is intentionally not machine
/// checkable, so it is absent here.
const FULL_FLOW_KNOBS: &[KnobCheck] = &[
    KnobCheck {
        knob: "max_infer_iters",
        source: KnobSource::Filter("openai_agentic_loop"),
        key: "max_infer_iters",
    },
    KnobCheck {
        knob: "max_iterations",
        source: KnobSource::Filter("iterative_request_router"),
        key: "max_iterations",
    },
    KnobCheck {
        knob: "timeout_ms",
        source: KnobSource::Filter("iterative_request_router"),
        key: "timeout_ms",
    },
    KnobCheck {
        knob: "step_timeout_ms",
        source: KnobSource::Filter("iterative_request_router"),
        key: "step_timeout_ms",
    },
    KnobCheck {
        knob: "max_response_bytes",
        source: KnobSource::Filter("iterative_request_router"),
        key: "max_response_bytes",
    },
    KnobCheck {
        knob: "max_stream_response_bytes",
        source: KnobSource::Filter("iterative_request_router"),
        key: "max_stream_response_bytes",
    },
    KnobCheck {
        knob: "max_state_bytes",
        source: KnobSource::Filter("iterative_request_router"),
        key: "max_state_bytes",
    },
    KnobCheck {
        knob: "file_search.timeout_ms",
        source: KnobSource::Filter("openai_file_search_callout"),
        key: "timeout_ms",
    },
    KnobCheck {
        knob: "file_search.max_response_bytes",
        source: KnobSource::Filter("openai_file_search_callout"),
        key: "max_response_bytes",
    },
    KnobCheck {
        knob: "file_search.max_total_response_bytes",
        source: KnobSource::Filter("openai_file_search_callout"),
        key: "max_total_response_bytes",
    },
    KnobCheck {
        knob: "file_search.max_state_bytes",
        source: KnobSource::Filter("openai_file_search_callout"),
        key: "max_state_bytes",
    },
    KnobCheck {
        knob: "file_search.on_failure",
        source: KnobSource::Filter("openai_file_search_callout"),
        key: "on_failure",
    },
    KnobCheck {
        knob: "mcp_dispatch.max_calls_per_round",
        source: KnobSource::Filter("openai_mcp_dispatch"),
        key: "max_calls_per_round",
    },
    KnobCheck {
        knob: "mcp_dispatch.max_parallel_calls",
        source: KnobSource::Filter("openai_mcp_dispatch"),
        key: "max_parallel_calls",
    },
    KnobCheck {
        knob: "mcp_dispatch.max_result_bytes",
        source: KnobSource::Filter("openai_mcp_dispatch"),
        key: "max_result_bytes",
    },
    KnobCheck {
        knob: "mcp_dispatch.max_total_result_bytes",
        source: KnobSource::Filter("openai_mcp_dispatch"),
        key: "max_total_result_bytes",
    },
    KnobCheck {
        knob: "web_search.max_calls_per_round",
        source: KnobSource::Filter("openai_web_search"),
        key: "max_calls_per_round",
    },
    KnobCheck {
        knob: "file_resolve.timeout_ms",
        source: KnobSource::Filter("openai_file_resolve"),
        key: "timeout_ms",
    },
    KnobCheck {
        knob: "file_resolve.on_missing",
        source: KnobSource::Filter("openai_file_resolve"),
        key: "on_missing",
    },
    KnobCheck {
        knob: "insecure_options.allow_private_endpoints",
        source: KnobSource::InsecureOptions,
        key: "allow_private_endpoints",
    },
    KnobCheck {
        knob: "insecure_options.allow_private_upstreams",
        source: KnobSource::InsecureOptions,
        key: "allow_private_upstreams",
    },
];

// -----------------------------------------------------------------------------
// CLI
// -----------------------------------------------------------------------------

/// CLI arguments for `cargo xtask sync-flow-visualizers`.
#[derive(Parser)]
pub(crate) struct Args {
    /// Reserved for HTML regeneration, added in a later increment. The current
    /// command validates structural drift in both modes; the visualizer HTML is
    /// still hand-maintained, so there is nothing to rewrite yet.
    #[arg(long)]
    fix: bool,
}

// -----------------------------------------------------------------------------
// Entry point
// -----------------------------------------------------------------------------

/// Validate every checked-in flow visualizer against its source config.
///
/// Exits with status 1 when any visualizer has drifted so `make lint` fails.
pub(crate) fn run(args: &Args) {
    if args.fix {
        println!("note: HTML regeneration is not implemented yet; validating structure only");
    }
    let root = workspace_root();
    let mut drifted = false;
    for viz in VISUALIZERS {
        match validate(&root, viz) {
            Ok(()) => println!("{}: structure matches {}", viz.html, viz.config),
            Err(errors) => {
                drifted = true;
                eprintln!(
                    "{}: {} structural difference(s) vs {}",
                    viz.html,
                    errors.len(),
                    viz.config
                );
                for error in &errors {
                    eprintln!("  - {error}");
                }
            },
        }
    }
    if drifted {
        eprintln!("update the visualizer HTML to match the config, then re-run");
        std::process::exit(1);
    }
}

// -----------------------------------------------------------------------------
// Validation
// -----------------------------------------------------------------------------

/// Validate one visualizer, returning every structural difference found.
///
/// # Errors
///
/// Returns the collected difference messages when the HTML has drifted from the
/// config, or a single-element vector describing an I/O or parse failure.
fn validate(root: &Path, viz: &Visualizer) -> Result<(), Vec<String>> {
    let graph = FlowGraph::from_file(&root.join(viz.config)).map_err(|err| vec![err])?;
    let nodes = graph
        .flatten_chain(viz.chain)
        .ok_or_else(|| vec![format!("chain `{}` not found in {}", viz.chain, viz.config)])?;
    let html = std::fs::read_to_string(root.join(viz.html)).map_err(|err| vec![format!("read {}: {err}", viz.html)])?;
    let pipeline = extract_json_literal(&html, "PIPELINE").map_err(|err| vec![err])?;

    let mut errors = Vec::new();
    check_filters(&pipeline, &nodes, &mut errors);
    check_clusters(&pipeline, &graph, &mut errors);
    check_knobs(&pipeline, &graph, &nodes, viz.knobs, &mut errors);
    if errors.is_empty() { Ok(()) } else { Err(errors) }
}

/// Check the filter sequence: count, 1-based order, and type at each position.
///
/// The type is recovered from the HTML display name, so this single pass fails
/// on any added, removed, reordered, or relocated filter (relocating a filter
/// into or out of the IRR changes its flattened index and therefore the
/// expected type at that index).
fn check_filters(pipeline: &Json, nodes: &[FlowNode], errors: &mut Vec<String>) {
    let Some(filters) = pipeline.get("filters").and_then(Json::as_array) else {
        errors.push("PIPELINE.filters is missing or not an array".to_owned());
        return;
    };
    if filters.len() != nodes.len() {
        errors.push(format!("filter count: HTML {}, config {}", filters.len(), nodes.len()));
    }
    for (index, node) in nodes.iter().enumerate() {
        let Some(filter) = filters.get(index) else {
            errors.push(format!(
                "config node {} (`{}`) has no HTML filter",
                node.order, node.filter_type
            ));
            continue;
        };
        check_filter_order(filter, node, index, errors);
        check_filter_type(filter, node, index, errors);
    }
    for extra in nodes.len()..filters.len() {
        let name = filters
            .get(extra)
            .and_then(|f| f.get("name"))
            .and_then(Json::as_str)
            .unwrap_or("?");
        errors.push(format!("HTML filter[{extra}] `{name}` has no config node"));
    }
}

/// Check that an HTML filter's `order` equals its 1-based position.
fn check_filter_order(filter: &Json, node: &FlowNode, index: usize, errors: &mut Vec<String>) {
    let html_order = filter.get("order").and_then(Json::as_u64);
    if html_order != u64::try_from(node.order).ok() {
        errors.push(format!(
            "filter[{index}] order: HTML {html_order:?}, config {}",
            node.order
        ));
    }
}

/// Check that the type recovered from an HTML filter name matches the config.
fn check_filter_type(filter: &Json, node: &FlowNode, index: usize, errors: &mut Vec<String>) {
    let name = filter.get("name").and_then(Json::as_str).unwrap_or("");
    let html_type = pipeline_type(name);
    if html_type != node.filter_type {
        errors.push(format!(
            "filter[{index}] type: HTML name `{name}` -> `{html_type}`, config `{}`",
            node.filter_type
        ));
    }
}

/// Recover a YAML filter type from an HTML display name by stripping any
/// ` (qualifier)` suffix (e.g. `"headers (IRR step)"` -> `"headers"`).
fn pipeline_type(name: &str) -> &str {
    name.split(" (").next().unwrap_or(name)
}

/// Check that every `load_balancer` cluster in the config is present in the HTML
/// with the same endpoint set, and vice versa.
fn check_clusters(pipeline: &Json, graph: &FlowGraph, errors: &mut Vec<String>) {
    let expected = config_clusters(graph);
    let actual = html_clusters(pipeline);
    for (name, endpoints) in &expected {
        match actual.get(name) {
            None => errors.push(format!("cluster `{name}` is in the config but missing from the HTML")),
            Some(html_endpoints) if html_endpoints != endpoints => errors.push(format!(
                "cluster `{name}` endpoints: HTML {html_endpoints:?}, config {endpoints:?}"
            )),
            Some(_) => {},
        }
    }
    for name in actual.keys() {
        if !expected.contains_key(name) {
            errors.push(format!("cluster `{name}` is in the HTML but not the config"));
        }
    }
}

/// Collect every `load_balancer` cluster declared anywhere in the config,
/// including inside branch chains and IRR steps, as `name -> endpoints`.
fn config_clusters(graph: &FlowGraph) -> BTreeMap<String, BTreeSet<String>> {
    let mut clusters = BTreeMap::new();
    for chain in &graph.filter_chains {
        for filter in &chain.filters {
            collect_clusters(&filter.config, &mut clusters);
            collect_clusters(&filter.branch_chains, &mut clusters);
        }
    }
    clusters
}

/// Recursively record every `clusters:` array found under a JSON value.
fn collect_clusters(value: &Json, out: &mut BTreeMap<String, BTreeSet<String>>) {
    match value {
        Json::Object(map) => {
            if let Some(Json::Array(list)) = map.get("clusters") {
                for entry in list {
                    record_cluster(entry, out);
                }
            }
            for nested in map.values() {
                collect_clusters(nested, out);
            }
        },
        Json::Array(items) => {
            for item in items {
                collect_clusters(item, out);
            }
        },
        _ => {},
    }
}

/// Record one cluster entry's name and endpoint strings.
fn record_cluster(entry: &Json, out: &mut BTreeMap<String, BTreeSet<String>>) {
    let Some(name) = entry.get("name").and_then(Json::as_str) else {
        return;
    };
    let set = out.entry(name.to_owned()).or_default();
    if let Some(Json::Array(endpoints)) = entry.get("endpoints") {
        for endpoint in endpoints {
            if let Some(text) = endpoint.as_str() {
                set.insert(text.to_owned());
            }
        }
    }
}

/// Build the `name -> endpoints` map declared in `PIPELINE.clusters`.
fn html_clusters(pipeline: &Json) -> BTreeMap<String, BTreeSet<String>> {
    let mut clusters = BTreeMap::new();
    if let Some(list) = pipeline.get("clusters").and_then(Json::as_array) {
        for entry in list {
            record_cluster(entry, &mut clusters);
        }
    }
    clusters
}

/// Check that every surfaced knob matches its authoritative config value.
fn check_knobs(pipeline: &Json, graph: &FlowGraph, nodes: &[FlowNode], knobs: &[KnobCheck], errors: &mut Vec<String>) {
    let html_knobs = pipeline.get("knobs");
    for check in knobs {
        let html_value = html_knobs.and_then(|knobs| knobs.get(check.knob));
        let config_value = knob_source_value(check, graph, nodes);
        match (html_value, config_value) {
            (Some(html), Some(config)) if html == config => {},
            (Some(html), Some(config)) => errors.push(format!("knob `{}`: HTML {html}, config {config}", check.knob)),
            (None, _) => errors.push(format!("knob `{}` is missing from the HTML", check.knob)),
            (_, None) => errors.push(format!(
                "knob `{}` has no value at its config source (key `{}`)",
                check.knob, check.key
            )),
        }
    }
}

/// Resolve a knob's authoritative value from its config source.
fn knob_source_value<'a>(check: &KnobCheck, graph: &'a FlowGraph, nodes: &'a [FlowNode]) -> Option<&'a Json> {
    match check.source {
        KnobSource::Filter(filter_type) => nodes
            .iter()
            .find(|node| node.filter_type == filter_type)
            .and_then(|node| node.config.get(check.key)),
        KnobSource::InsecureOptions => graph.insecure_options.get(check.key),
    }
}

// -----------------------------------------------------------------------------
// HTML data extraction
// -----------------------------------------------------------------------------

/// Extract a `const <name> = { ... };` JSON object literal from the HTML.
///
/// The literal is located by its `const` declaration and sliced with a
/// brace/bracket balancer that respects string state, then parsed as strict
/// JSON (the visualizer's data blocks are emitted as JSON).
///
/// # Errors
///
/// Returns an error when the declaration is absent, the literal is unbalanced,
/// or the sliced text is not valid JSON.
fn extract_json_literal(html: &str, name: &str) -> Result<Json, String> {
    let anchor = format!("const {name}");
    let decl = html
        .find(&anchor)
        .ok_or_else(|| format!("`const {name}` not found in the HTML"))?;
    let tail = html.get(decl + anchor.len()..).unwrap_or_default();
    let open_offset = tail
        .find(['{', '['])
        .ok_or_else(|| format!("no opening brace after `const {name}`"))?;
    let open = decl + anchor.len() + open_offset;
    let end = balanced_end(html, open).ok_or_else(|| format!("unbalanced literal for `const {name}`"))?;
    let literal = html
        .get(open..end)
        .ok_or_else(|| format!("could not slice literal for `const {name}`"))?;
    serde_json::from_str(literal).map_err(|err| format!("parse `const {name}` as JSON: {err}"))
}

/// Return the byte index one past the balanced close of the bracket at `open`,
/// respecting quoted strings and escapes. Returns `None` when unbalanced.
fn balanced_end(html: &str, open: usize) -> Option<usize> {
    let bytes = html.as_bytes();
    let open_ch = *bytes.get(open)?;
    let close_ch = if open_ch == b'{' { b'}' } else { b']' };
    let mut depth = 0_i32;
    let mut in_string = false;
    let mut escaped = false;
    let mut quote = 0_u8;
    for (index, &byte) in bytes.iter().enumerate().skip(open) {
        if escaped {
            escaped = false;
        } else if in_string && byte == b'\\' {
            escaped = true;
        } else if in_string {
            in_string = byte != quote;
        } else if matches!(byte, b'"' | b'\'' | b'`') {
            in_string = true;
            quote = byte;
        } else if byte == open_ch {
            depth += 1;
        } else if byte == close_ch {
            depth -= 1;
            if depth == 0 {
                return Some(index + 1);
            }
        }
    }
    None
}

// -----------------------------------------------------------------------------
// Paths
// -----------------------------------------------------------------------------

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

    fn full_flow() -> &'static Visualizer {
        VISUALIZERS.first().expect("the full-flow visualizer is registered")
    }

    #[test]
    fn full_flow_visualizer_has_no_structural_drift() {
        let root = workspace_root();
        validate(&root, full_flow()).expect("committed HTML must match the committed config");
    }

    #[test]
    fn pipeline_type_strips_qualifier_suffix() {
        assert_eq!(pipeline_type("headers (IRR step)"), "headers");
        assert_eq!(pipeline_type("router (IRR step)"), "router");
        assert_eq!(pipeline_type("openai_responses_proxy"), "openai_responses_proxy");
    }

    #[test]
    fn extract_json_literal_reads_the_pipeline_block() {
        let root = workspace_root();
        let html = std::fs::read_to_string(root.join(full_flow().html)).expect("HTML readable");
        let pipeline = extract_json_literal(&html, "PIPELINE").expect("PIPELINE parses");
        let filters = pipeline.get("filters").and_then(Json::as_array).expect("filters array");
        assert_eq!(filters.len(), 25, "15 main-chain + 10 IRR-step filters");
    }

    #[test]
    fn missing_filter_is_flagged() {
        let graph = FlowGraph::from_yaml_str(&minimal("trace_context", "state_owner")).expect("valid");
        let nodes = graph.flatten_chain("c").expect("chain");
        let pipeline = serde_json::json!({
            "filters": [{ "order": 1, "name": "trace_context" }],
        });
        let mut errors = Vec::new();
        check_filters(&pipeline, &nodes, &mut errors);
        assert!(
            errors.iter().any(|e| e.contains("filter count")),
            "a dropped filter must be reported, got {errors:?}"
        );
    }

    #[test]
    fn reordered_filter_type_is_flagged() {
        let graph = FlowGraph::from_yaml_str(&minimal("trace_context", "state_owner")).expect("valid");
        let nodes = graph.flatten_chain("c").expect("chain");
        let pipeline = serde_json::json!({
            "filters": [
                { "order": 1, "name": "state_owner" },
                { "order": 2, "name": "trace_context" },
            ],
        });
        let mut errors = Vec::new();
        check_filters(&pipeline, &nodes, &mut errors);
        assert!(
            errors.iter().any(|e| e.contains("type")),
            "a swapped filter type must be reported, got {errors:?}"
        );
    }

    #[test]
    fn changed_cluster_endpoint_is_flagged() {
        let graph = FlowGraph::from_yaml_str(&minimal_load_balancer("10.0.0.1:1")).expect("valid");
        let pipeline = serde_json::json!({
            "clusters": [{ "name": "c1", "endpoints": ["10.0.0.9:9"] }],
        });
        let mut errors = Vec::new();
        check_clusters(&pipeline, &graph, &mut errors);
        assert!(
            errors.iter().any(|e| e.contains("endpoints")),
            "a changed endpoint must be reported, got {errors:?}"
        );
    }

    /// A minimal valid config: a listener plus a chain `c` of two named filters.
    fn minimal(first: &str, second: &str) -> String {
        format!(
            "listeners:\n  - name: l\n    address: 127.0.0.1:8080\n    filter_chains: [c]\n\
             filter_chains:\n  - name: c\n    filters:\n      - filter: {first}\n      - filter: {second}\n"
        )
    }

    /// A minimal valid config whose chain ends in a `load_balancer` with one
    /// cluster `c1` bound to `endpoint`.
    fn minimal_load_balancer(endpoint: &str) -> String {
        format!(
            "listeners:\n  - name: l\n    address: 127.0.0.1:8080\n    filter_chains: [c]\n\
             filter_chains:\n  - name: c\n    filters:\n      - filter: router\n        routes:\n\
             \x20         - path: /\n            cluster: c1\n      - filter: load_balancer\n        clusters:\n\
             \x20         - name: c1\n            endpoints:\n              - \"{endpoint}\"\n"
        )
    }
}
