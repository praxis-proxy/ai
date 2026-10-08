// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Deterministic flow-visualizer renderer.
//!
//! [`render`] merges the mechanically inferred structure of a config
//! (via [`crate::flow_graph::FlowGraph`]) with the curated narration held in a
//! visualizer sidecar, and substitutes the result into a static HTML template.
//! The split is deliberate and enforced:
//!
//! * **Structure** — filter order, presence, and type; every `load_balancer` cluster and its endpoints; every scalar
//!   limit surfaced as a knob — is sourced from, and cross-checked against, the parsed config. Any drift is returned as
//!   an actionable error rather than silently rendered.
//! * **Semantics** — synthetic ids, display names, groups, phase/promotes prose, per-scenario notes, wire payloads,
//!   timelines, and cluster/service annotations — come from the sidecar verbatim.
//!
//! Every data block is emitted as order-preserving pretty JSON so regeneration
//! is byte-for-byte reproducible and diffs stay legible.

use std::collections::{BTreeMap, BTreeSet};

use serde_json::Value as JsonV;
use serde_yaml::{Mapping, Number, Value};

use crate::{
    flow_graph::{FlowGraph, FlowNode, IRR_FILTER},
    html,
};

/// A shared empty value returned when a sidecar block is absent (the caller has
/// already recorded the error, so the emitted placeholder is never used).
static NULL: Value = Value::Null;

/// The curated data blocks emitted verbatim from the sidecar, paired with the
/// template placeholder each one fills.
const VERBATIM_BLOCKS: [(&str, &str); 5] = [
    ("scen_meta", "@@SCEN_META@@"),
    ("wire", "@@WIRE@@"),
    ("timeline", "@@TIMELINE@@"),
    ("groups", "@@GROUPS@@"),
    ("state_label", "@@STATE_LABEL@@"),
];

// -----------------------------------------------------------------------------
// Knob registry types
// -----------------------------------------------------------------------------

/// A scalar limit surfaced in the visualizer `knobs` map and the config value it
/// must match.
pub(crate) struct KnobCheck {
    /// Key in the sidecar `pipeline.knobs` object (and the rendered HTML).
    pub(crate) knob: &'static str,
    /// Where the authoritative value lives in the parsed config.
    pub(crate) source: KnobSource,
    /// Config key to read within that source.
    pub(crate) key: &'static str,
}

/// The config location that owns a knob's authoritative value.
pub(crate) enum KnobSource {
    /// A filter's opaque config, located by its (unique) filter type.
    Filter(&'static str),
    /// The global `insecure_options` block.
    InsecureOptions,
}

// -----------------------------------------------------------------------------
// Rendering
// -----------------------------------------------------------------------------

/// Render the visualizer HTML for one config + sidecar + template.
///
/// # Errors
///
/// Returns every structural disagreement found between the parsed config and the
/// sidecar (missing/extra/misordered/mistyped filters, cluster or knob drift, or
/// missing sidecar blocks). The HTML is produced only when no errors remain.
pub(crate) fn render(
    graph: &FlowGraph,
    chain: &str,
    sidecar: &Value,
    template: &str,
    knobs: &[KnobCheck],
) -> Result<String, Vec<String>> {
    let nodes = graph
        .flatten_chain(chain)
        .ok_or_else(|| vec![format!("chain not found in config: {chain}")])?;
    let mut errors = Vec::new();
    let pipeline = build_pipeline(&nodes, sidecar, graph, knobs, &mut errors);
    let blocks: Vec<(&str, &Value)> = VERBATIM_BLOCKS
        .iter()
        .map(|(key, _)| (*key, block(sidecar, key, &mut errors)))
        .collect();
    if !errors.is_empty() {
        return Err(errors);
    }
    // Every block is inlined into a <script> as `const X = {...};`, so any
    // `</script>` (or `<!--`) inside a curated string must be neutralized or it
    // would break the page. Escaping leaves the parsed JSON identical.
    let mut out = template.replace(
        "@@PIPELINE@@",
        &html::escape_json_for_script(&to_pretty_json(&pipeline)),
    );
    for ((_, placeholder), (_, value)) in VERBATIM_BLOCKS.iter().zip(blocks.iter()) {
        out = out.replace(placeholder, &html::escape_json_for_script(&to_pretty_json(value)));
    }
    Ok(out)
}

/// Look up a top-level sidecar block, recording an error when it is absent.
fn block<'a>(sidecar: &'a Value, key: &str, errors: &mut Vec<String>) -> &'a Value {
    if let Some(value) = sidecar.get(key) {
        value
    } else {
        errors.push(format!("sidecar block missing: {key}"));
        &NULL
    }
}

// -----------------------------------------------------------------------------
// PIPELINE assembly
// -----------------------------------------------------------------------------

/// Assemble the `PIPELINE` value: structural order from the flattened config,
/// curated fields from the sidecar, with topology/cluster/knob cross-checks.
fn build_pipeline(
    nodes: &[FlowNode],
    sidecar: &Value,
    graph: &FlowGraph,
    knobs: &[KnobCheck],
    errors: &mut Vec<String>,
) -> Value {
    let pipeline = sidecar.get("pipeline");
    let filters = align_filters(nodes, pipeline, errors);
    check_clusters(graph, pipeline, errors);
    check_knobs(nodes, graph, pipeline, knobs, errors);

    let mut map = Mapping::new();
    map.insert(Value::from("filters"), Value::Sequence(filters));
    map.insert(Value::from("structure"), build_structure(nodes));
    for key in ["scenarios", "clusters", "externalServices", "knobs"] {
        let value = pipeline.and_then(|p| p.get(key)).cloned().unwrap_or_else(|| {
            errors.push(format!("sidecar.pipeline.{key} missing"));
            Value::Null
        });
        map.insert(Value::from(key), value);
    }
    Value::Mapping(map)
}

/// Bind each flattened config node to its sidecar filter entry, reporting
/// count/type/order drift, and emit the merged filter objects.
fn align_filters(nodes: &[FlowNode], pipeline: Option<&Value>, errors: &mut Vec<String>) -> Vec<Value> {
    let Some(list) = pipeline.and_then(|p| p.get("filters")).and_then(Value::as_sequence) else {
        errors.push("sidecar.pipeline.filters missing or not a sequence".to_owned());
        return Vec::new();
    };
    if list.len() != nodes.len() {
        errors.push(format!(
            "filter count: config has {}, sidecar has {}",
            nodes.len(),
            list.len()
        ));
    }
    let mut out = Vec::with_capacity(nodes.len());
    for (index, node) in nodes.iter().enumerate() {
        let Some(entry) = list.get(index) else { continue };
        check_filter_type(entry, node, index, errors);
        out.push(merge_filter(entry, index + 1));
    }
    report_extra_filters(list, nodes.len(), errors);
    out
}

/// Report a type mismatch between a sidecar filter entry and its config node.
fn check_filter_type(entry: &Value, node: &FlowNode, index: usize, errors: &mut Vec<String>) {
    let declared = filter_type_of(entry);
    if declared != node.filter_type {
        errors.push(format!(
            "filter {}: config type {:?} != sidecar type {declared:?}",
            index + 1,
            node.filter_type
        ));
    }
}

/// Report each sidecar filter beyond the last config node as unmatched.
fn report_extra_filters(list: &[Value], start: usize, errors: &mut Vec<String>) {
    for (index, entry) in list.iter().enumerate().skip(start) {
        errors.push(format!(
            "filter {}: sidecar type {:?} has no matching config filter",
            index + 1,
            filter_type_of(entry)
        ));
    }
}

/// The `filter_type` binding key of a sidecar filter entry, or `""` if absent.
fn filter_type_of(entry: &Value) -> &str {
    entry.get("filter_type").and_then(Value::as_str).unwrap_or_default()
}

/// Build one PIPELINE filter object in the canonical field order, taking `order`
/// from the flattened config and every other field from the sidecar entry.
fn merge_filter(entry: &Value, order: usize) -> Value {
    let mut map = Mapping::new();
    for key in ["id", "name", "group"] {
        map.insert(Value::from(key), entry.get(key).cloned().unwrap_or(Value::Null));
    }
    let order = u64::try_from(order).unwrap_or_default();
    map.insert(Value::from("order"), Value::Number(Number::from(order)));
    for key in ["phase", "config", "promotes", "scenarios"] {
        map.insert(Value::from(key), entry.get(key).cloned().unwrap_or(Value::Null));
    }
    Value::Mapping(map)
}

// -----------------------------------------------------------------------------
// Structural signature (config-authoritative topology)
// -----------------------------------------------------------------------------

/// Build the config-authoritative `structure` block emitted alongside the curated
/// pipeline.
///
/// The curated cross-checks above cover filter count/type/order, clusters, and
/// the registered scalar knobs — but not a filter's `conditions`, its
/// `branch_chains` (names, rejoin targets, nested routes and filters), a
/// `router`'s `routes`, or the IRR's `initial_step` and step transitions. Because
/// the generated HTML is verified byte-for-byte by `sync-flow-visualizers`,
/// embedding this mechanically inferred fingerprint makes any such topology drift
/// force a regeneration, so no structural change lands silently. It is derived
/// only from the parsed config, never the sidecar.
fn build_structure(nodes: &[FlowNode]) -> Value {
    let filters: Vec<Value> = nodes.iter().map(structure_node).collect();
    let mut map = Mapping::new();
    map.insert(Value::from("filters"), Value::Sequence(filters));
    if let Some(irr) = irr_routing(nodes) {
        map.insert(Value::from("irr"), irr);
    }
    Value::Mapping(map)
}

/// Capture one flattened node's structural identity plus every topology-bearing
/// field: its `conditions`, `branch_chains`, and any `routes` in its config.
fn structure_node(node: &FlowNode) -> Value {
    let mut map = Mapping::new();
    map.insert(Value::from("order"), number(node.order));
    map.insert(Value::from("depth"), number(node.depth));
    map.insert(Value::from("filter_type"), Value::from(node.filter_type.as_str()));
    map.insert(Value::from("name"), opt_string(node.name.as_deref()));
    map.insert(Value::from("irr_step"), opt_string(node.irr_step.as_deref()));
    map.insert(Value::from("conditions"), json_to_yaml(&node.conditions));
    map.insert(Value::from("branch_chains"), json_to_yaml(&node.branch_chains));
    if let Some(routes) = node.config.get("routes") {
        map.insert(Value::from("routes"), json_to_yaml(routes));
    }
    Value::Mapping(map)
}

/// Capture the `iterative_request_router` step-transition graph: the
/// `initial_step` and, per step, its name plus every routing key other than the
/// (already-flattened) `filters` list.
fn irr_routing(nodes: &[FlowNode]) -> Option<Value> {
    let router = nodes.iter().find(|node| node.filter_type == IRR_FILTER)?;
    let config = router.config.as_object()?;
    let mut map = Mapping::new();
    if let Some(initial) = config.get("initial_step") {
        map.insert(Value::from("initial_step"), json_to_yaml(initial));
    }
    if let Some(steps) = config.get("steps").and_then(JsonV::as_array) {
        let summaries: Vec<Value> = steps.iter().filter_map(step_routing).collect();
        map.insert(Value::from("steps"), Value::Sequence(summaries));
    }
    if map.is_empty() {
        None
    } else {
        Some(Value::Mapping(map))
    }
}

/// Summarize one IRR step: every key except the already-flattened `filters` list,
/// so transitions (`on_result`, `next`, `default`, `done`, conditions) are kept.
fn step_routing(step: &JsonV) -> Option<Value> {
    let object = step.as_object()?;
    let mut map = Mapping::new();
    for (key, value) in object {
        if key != "filters" {
            map.insert(Value::from(key.as_str()), json_to_yaml(value));
        }
    }
    Some(Value::Mapping(map))
}

/// A `serde_yaml` unsigned number from a `usize` position or depth.
fn number(value: usize) -> Value {
    Value::Number(Number::from(u64::try_from(value).unwrap_or_default()))
}

/// A YAML string, or null when the optional slice is absent.
fn opt_string(value: Option<&str>) -> Value {
    value.map_or(Value::Null, Value::from)
}

/// Convert a `serde_json` value into the `serde_yaml` value the emitter expects.
fn json_to_yaml(value: &JsonV) -> Value {
    serde_yaml::to_value(value).unwrap_or(Value::Null)
}

// -----------------------------------------------------------------------------
// Cluster cross-check
// -----------------------------------------------------------------------------

/// Fail when the sidecar's declared clusters disagree with the `load_balancer`
/// clusters the config actually declares (however deeply nested).
fn check_clusters(graph: &FlowGraph, pipeline: Option<&Value>, errors: &mut Vec<String>) {
    let expected = config_clusters(graph);
    let actual = sidecar_clusters(pipeline);
    for (name, endpoints) in &expected {
        match actual.get(name) {
            None => errors.push(format!("cluster {name}: declared in config, missing from sidecar")),
            Some(found) if found != endpoints => errors.push(format!(
                "cluster {name}: config endpoints {endpoints:?} != sidecar {found:?}"
            )),
            Some(_) => {},
        }
    }
    for name in actual.keys() {
        if !expected.contains_key(name) {
            errors.push(format!("cluster {name}: declared in sidecar, missing from config"));
        }
    }
}

/// Collect every `load_balancer` cluster the config declares, keyed by name to
/// its set of endpoints.
fn config_clusters(graph: &FlowGraph) -> BTreeMap<String, BTreeSet<String>> {
    let mut out = BTreeMap::new();
    let value = serde_json::to_value(graph).unwrap_or(JsonV::Null);
    collect_clusters(&value, &mut out);
    out
}

/// Recursively record clusters found in any `clusters` array within `value`.
fn collect_clusters(value: &JsonV, out: &mut BTreeMap<String, BTreeSet<String>>) {
    match value {
        JsonV::Object(map) => {
            if let Some(JsonV::Array(list)) = map.get("clusters") {
                for cluster in list {
                    record_cluster(cluster, out);
                }
            }
            for nested in map.values() {
                collect_clusters(nested, out);
            }
        },
        JsonV::Array(items) => {
            for item in items {
                collect_clusters(item, out);
            }
        },
        _ => {},
    }
}

/// Record one cluster's name and endpoint set when both are present.
fn record_cluster(cluster: &JsonV, out: &mut BTreeMap<String, BTreeSet<String>>) {
    let Some(name) = cluster.get("name").and_then(JsonV::as_str) else {
        return;
    };
    let Some(endpoints) = cluster.get("endpoints").and_then(JsonV::as_array) else {
        return;
    };
    let set: BTreeSet<String> = endpoints.iter().filter_map(JsonV::as_str).map(str::to_owned).collect();
    out.entry(name.to_owned()).or_default().extend(set);
}

/// Collect the clusters the sidecar declares in `pipeline.clusters`.
fn sidecar_clusters(pipeline: Option<&Value>) -> BTreeMap<String, BTreeSet<String>> {
    let mut out = BTreeMap::new();
    let Some(list) = pipeline.and_then(|p| p.get("clusters")).and_then(Value::as_sequence) else {
        return out;
    };
    for cluster in list {
        let Some(name) = cluster.get("name").and_then(Value::as_str) else {
            continue;
        };
        let Some(endpoints) = cluster.get("endpoints").and_then(Value::as_sequence) else {
            continue;
        };
        let set: BTreeSet<String> = endpoints.iter().filter_map(Value::as_str).map(str::to_owned).collect();
        out.entry(name.to_owned()).or_default().extend(set);
    }
    out
}

// -----------------------------------------------------------------------------
// Knob cross-check
// -----------------------------------------------------------------------------

/// Fail when any sidecar knob disagrees with its authoritative config source.
fn check_knobs(
    nodes: &[FlowNode],
    graph: &FlowGraph,
    pipeline: Option<&Value>,
    knobs: &[KnobCheck],
    errors: &mut Vec<String>,
) {
    let sidecar = pipeline.and_then(|p| p.get("knobs"));
    for check in knobs {
        let expected = knob_expected(nodes, graph, check);
        let actual = sidecar.and_then(|k| k.get(check.knob)).map(to_json);
        match (expected, actual) {
            (Some(exp), Some(act)) if exp == act => {},
            (Some(exp), Some(act)) => {
                errors.push(format!("knob {}: config {exp} != sidecar {act}", check.knob));
            },
            (Some(exp), None) => errors.push(format!("knob {}: config {exp} but missing from sidecar", check.knob)),
            (None, _) => errors.push(format!("knob {}: no value found at config source", check.knob)),
        }
    }
}

/// Resolve a knob's authoritative value from the parsed config.
fn knob_expected(nodes: &[FlowNode], graph: &FlowGraph, check: &KnobCheck) -> Option<JsonV> {
    match check.source {
        KnobSource::Filter(filter_type) => nodes
            .iter()
            .find(|node| node.filter_type == filter_type)
            .and_then(|node| node.config.get(check.key).cloned()),
        KnobSource::InsecureOptions => graph.insecure_options.get(check.key).cloned(),
    }
}

// -----------------------------------------------------------------------------
// Order-preserving pretty JSON
// -----------------------------------------------------------------------------

/// Serialize a YAML value as pretty (2-space) JSON, preserving mapping order.
/// The output matches `JSON.stringify(value, null, 2)` for the shapes used here.
fn to_pretty_json(value: &Value) -> String {
    let mut out = String::new();
    write_value(&mut out, value, 0);
    out
}

/// Write any value at the given indent depth.
fn write_value(out: &mut String, value: &Value, indent: usize) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(flag) => out.push_str(if *flag { "true" } else { "false" }),
        Value::Number(number) => out.push_str(&number.to_string()),
        Value::String(text) => write_json_string(out, text),
        Value::Sequence(items) => write_array(out, items, indent),
        Value::Mapping(map) => write_object(out, map, indent),
        Value::Tagged(tagged) => write_value(out, &tagged.value, indent),
    }
}

/// Write a JSON array, one element per line.
fn write_array(out: &mut String, items: &[Value], indent: usize) {
    if items.is_empty() {
        out.push_str("[]");
        return;
    }
    out.push('[');
    for (index, item) in items.iter().enumerate() {
        if index > 0 {
            out.push(',');
        }
        out.push('\n');
        push_indent(out, indent + 1);
        write_value(out, item, indent + 1);
    }
    out.push('\n');
    push_indent(out, indent);
    out.push(']');
}

/// Write a JSON object, one entry per line, preserving mapping order.
fn write_object(out: &mut String, map: &Mapping, indent: usize) {
    if map.is_empty() {
        out.push_str("{}");
        return;
    }
    out.push('{');
    for (index, (key, value)) in map.iter().enumerate() {
        if index > 0 {
            out.push(',');
        }
        out.push('\n');
        push_indent(out, indent + 1);
        write_json_string(out, key.as_str().unwrap_or_default());
        out.push_str(": ");
        write_value(out, value, indent + 1);
    }
    out.push('\n');
    push_indent(out, indent);
    out.push('}');
}

/// Write a JSON string literal, escaping per RFC 8259 and keeping UTF-8 literal.
fn write_json_string(out: &mut String, text: &str) {
    out.push('"');
    for ch in text.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            control if (control as u32) < 0x20 => push_control_escape(out, control),
            other => out.push(other),
        }
    }
    out.push('"');
}

/// Append a `\u00xx` escape for a control character below `0x20`.
fn push_control_escape(out: &mut String, control: char) {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let code = control as usize;
    out.push_str("\\u00");
    out.push(char::from(*HEX.get((code >> 4) & 0xF).unwrap_or(&b'0')));
    out.push(char::from(*HEX.get(code & 0xF).unwrap_or(&b'0')));
}

/// Push `indent` levels of two-space indentation.
fn push_indent(out: &mut String, indent: usize) {
    for _ in 0..indent {
        out.push_str("  ");
    }
}

/// Convert a YAML value into a JSON value for structural comparison.
fn to_json(value: &Value) -> JsonV {
    serde_json::to_value(value).unwrap_or(JsonV::Null)
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing, reason = "tests")]
mod tests {
    use super::*;

    /// A minimal config: a listener plus a chain `c` of two named filters.
    const CONFIG: &str = "listeners:\n  - name: l\n    address: 127.0.0.1:8080\n    filter_chains: [c]\n\
        filter_chains:\n  - name: c\n    filters:\n      - filter: trace_context\n      - filter: state_owner\n";

    /// A sidecar that aligns with [`CONFIG`], with all blocks present.
    const SIDECAR: &str = "pipeline:\n  filters:\n\
        \x20   - id: a\n      name: trace_context\n      group: g\n      filter_type: trace_context\n\
        \x20     phase: p\n      config: {}\n      promotes: []\n      scenarios: {}\n\
        \x20   - id: b\n      name: state_owner\n      group: g\n      filter_type: state_owner\n\
        \x20     phase: p\n      config: {}\n      promotes: []\n      scenarios: {}\n\
        \x20 scenarios: []\n  clusters: []\n  externalServices: []\n  knobs: {}\n\
        scen_meta: {}\nwire: {}\ntimeline: {}\ngroups: {}\nstate_label: {}\n";

    /// A template with every block placeholder, separated so replacement is
    /// observable.
    const TEMPLATE: &str = "@@PIPELINE@@|@@SCEN_META@@|@@WIRE@@|@@TIMELINE@@|@@GROUPS@@|@@STATE_LABEL@@";

    fn graph() -> FlowGraph {
        FlowGraph::from_yaml_str(CONFIG).expect("config parses")
    }

    fn sidecar() -> Value {
        serde_yaml::from_str(SIDECAR).expect("sidecar parses")
    }

    #[test]
    fn renders_when_config_and_sidecar_align() {
        let html = render(&graph(), "c", &sidecar(), TEMPLATE, &[]).expect("aligned inputs render");
        assert!(!html.contains("@@"), "all placeholders substituted");
        assert!(html.contains("\"trace_context\""), "filter surfaced in PIPELINE");
        assert!(html.contains("\"order\": 1"), "order sourced from the config");
    }

    #[test]
    fn reports_filter_type_drift() {
        let mut side = sidecar();
        *side
            .get_mut("pipeline")
            .and_then(|p| p.get_mut("filters"))
            .and_then(|f| f.get_mut(1))
            .and_then(|e| e.get_mut("filter_type"))
            .unwrap() = Value::from("wrong_type");
        let errors = render(&graph(), "c", &side, TEMPLATE, &[]).expect_err("type drift must fail");
        assert!(errors.iter().any(|e| e.contains("filter 2")), "got {errors:?}");
    }

    #[test]
    fn reports_reordered_filters() {
        let mut side = sidecar();
        let filters = side.get_mut("pipeline").and_then(|p| p.get_mut("filters")).unwrap();
        if let Value::Sequence(seq) = filters {
            seq.reverse();
        }
        let errors = render(&graph(), "c", &side, TEMPLATE, &[]).expect_err("reorder must fail");
        assert!(errors.iter().any(|e| e.contains("filter")), "got {errors:?}");
    }

    #[test]
    fn reports_filter_count_drift() {
        let mut side = sidecar();
        let filters = side.get_mut("pipeline").and_then(|p| p.get_mut("filters")).unwrap();
        if let Value::Sequence(seq) = filters {
            let extra = seq.last().cloned().unwrap();
            seq.push(extra);
        }
        let errors = render(&graph(), "c", &side, TEMPLATE, &[]).expect_err("count drift must fail");
        assert!(errors.iter().any(|e| e.contains("filter count")), "got {errors:?}");
        assert!(
            errors.iter().any(|e| e.contains("no matching config filter")),
            "got {errors:?}"
        );
    }

    #[test]
    fn reports_missing_sidecar_block() {
        let mut side = sidecar();
        if let Value::Mapping(map) = &mut side {
            map.remove(Value::from("wire"));
        }
        let errors = render(&graph(), "c", &side, TEMPLATE, &[]).expect_err("missing block must fail");
        assert!(errors.iter().any(|e| e.contains("wire")), "got {errors:?}");
    }

    #[test]
    fn reports_malformed_sidecar_filters_not_a_sequence() {
        let mut side = sidecar();
        // A malformed sidecar whose pipeline.filters is a mapping, not the
        // required sequence, must fail cleanly rather than panic or silently
        // render an empty pipeline.
        *side.get_mut("pipeline").and_then(|p| p.get_mut("filters")).unwrap() = Value::Mapping(Mapping::new());
        let errors = render(&graph(), "c", &side, TEMPLATE, &[]).expect_err("malformed filters must fail");
        assert!(errors.iter().any(|e| e.contains("not a sequence")), "got {errors:?}");
    }

    #[test]
    fn reports_knob_drift() {
        let config = "listeners:\n  - name: l\n    address: 127.0.0.1:8080\n    filter_chains: [c]\n\
            filter_chains:\n  - name: c\n    filters:\n      - filter: trace_context\n\
            \x20     - filter: openai_agentic_loop\n        max_infer_iters: 4\n";
        let graph = FlowGraph::from_yaml_str(config).expect("config parses");
        let side: Value = serde_yaml::from_str(
            "pipeline:\n  filters:\n\
             \x20   - id: a\n      name: trace_context\n      group: g\n      filter_type: trace_context\n\
             \x20     phase: p\n      config: {}\n      promotes: []\n      scenarios: {}\n\
             \x20   - id: b\n      name: openai_agentic_loop\n      group: g\n      filter_type: openai_agentic_loop\n\
             \x20     phase: p\n      config: {}\n      promotes: []\n      scenarios: {}\n\
             \x20 scenarios: []\n  clusters: []\n  externalServices: []\n  knobs:\n    max_infer_iters: 9\n\
             scen_meta: {}\nwire: {}\ntimeline: {}\ngroups: {}\nstate_label: {}\n",
        )
        .expect("sidecar parses");
        let knobs = [KnobCheck {
            knob: "max_infer_iters",
            source: KnobSource::Filter("openai_agentic_loop"),
            key: "max_infer_iters",
        }];
        let errors = render(&graph, "c", &side, TEMPLATE, &knobs).expect_err("knob drift must fail");
        assert!(errors.iter().any(|e| e.contains("max_infer_iters")), "got {errors:?}");
    }

    #[test]
    fn reports_cluster_endpoint_drift() {
        let config = "listeners:\n  - name: l\n    address: 127.0.0.1:8080\n    filter_chains: [c]\n\
            filter_chains:\n  - name: c\n    filters:\n      - filter: router\n        routes:\n\
            \x20         - path: /\n            cluster: c1\n      - filter: load_balancer\n        clusters:\n\
            \x20         - name: c1\n            endpoints:\n              - \"10.0.0.1:1\"\n";
        let graph = FlowGraph::from_yaml_str(config).expect("config parses");
        let side: Value = serde_yaml::from_str(
            "pipeline:\n  filters:\n\
             \x20   - id: a\n      name: router\n      group: g\n      filter_type: router\n\
             \x20     phase: p\n      config: {}\n      promotes: []\n      scenarios: {}\n\
             \x20   - id: b\n      name: load_balancer\n      group: g\n      filter_type: load_balancer\n\
             \x20     phase: p\n      config: {}\n      promotes: []\n      scenarios: {}\n\
             \x20 scenarios: []\n  clusters:\n    - name: c1\n      endpoints:\n        - \"10.0.0.9:9\"\n\
             \x20 externalServices: []\n  knobs: {}\n\
             scen_meta: {}\nwire: {}\ntimeline: {}\ngroups: {}\nstate_label: {}\n",
        )
        .expect("sidecar parses");
        let errors = render(&graph, "c", &side, TEMPLATE, &[]).expect_err("cluster drift must fail");
        assert!(errors.iter().any(|e| e.contains("cluster c1")), "got {errors:?}");
    }

    #[test]
    fn pretty_json_is_valid_and_faithful() {
        let value: Value =
            serde_yaml::from_str("a: 1\nb:\n  - x\n  - \"line\\nbreak\"\nc: {}\nd: []\ne: true\nf: null\ng: 67108864")
                .expect("value parses");
        let json = to_pretty_json(&value);
        let parsed: JsonV = serde_json::from_str(&json).expect("emitted text is valid JSON");
        assert_eq!(parsed, to_json(&value), "pretty JSON must round-trip losslessly");
    }

    // -------------------------------------------------------------------------
    // Structural signature drift
    // -------------------------------------------------------------------------

    /// A config faithful to the full-flow topology that the sidecar cross-checks
    /// cannot see: a bypass carrier with an `unless` condition and a branch chain
    /// (its own router routes plus a `load_balancer`), and an IRR whose step carries
    /// a router and an `on_result` transition graph. Each drift test mutates a
    /// single, uniquely spelled token of this base.
    const TOPOLOGY_CONFIG: &str = r#"
listeners:
  - name: l
    address: 127.0.0.1:8080
    filter_chains: [c]
filter_chains:
  - name: c
    filters:
      - filter: headers
        conditions:
          - unless:
              path: "/v1/responses"
              methods: [POST]
        branch_chains:
          - name: bypass-irr
            rejoin: terminal
            chains:
              - name: bypass-chain
                filters:
                  - filter: router
                    routes:
                      - path_prefix: "/v1/files"
                        cluster: files-api
                  - filter: load_balancer
                    clusters:
                      - name: files-api
                        endpoints: ["127.0.0.1:9999"]
      - filter: iterative_request_router
        initial_step: inference
        max_iterations: 4
        steps:
          - name: inference
            filters:
              - filter: openai_agentic_loop
                max_infer_iters: 3
              - filter: router
                routes:
                  - path: "/v1/inference"
                    cluster: inference-backend
              - filter: load_balancer
                clusters:
                  - name: inference-backend
                    endpoints: ["127.0.0.1:3001"]
              - filter: openai_responses_proxy
            on_result:
              - filter: openai_agentic_loop
                key: action
                value: loop
                next: inference
              - default: true
                done: true
insecure_options:
  allow_private_endpoints: true
"#;

    /// Flatten [`TOPOLOGY_CONFIG`] (or a mutated variant) and emit its structural
    /// signature as pretty JSON — exactly the text embedded in the visualizer.
    fn structure_json(config: &str) -> String {
        let graph = FlowGraph::from_yaml_str(config).expect("topology config parses");
        let nodes = graph.flatten_chain("c").expect("chain c exists");
        to_pretty_json(&build_structure(&nodes))
    }

    #[test]
    fn structure_captures_branch_and_irr_topology() {
        let json = structure_json(TOPOLOGY_CONFIG);
        // Branch topology the curated cross-checks never inspect.
        assert!(json.contains("bypass-irr"), "branch name captured: {json}");
        assert!(json.contains("\"rejoin\": \"terminal\""), "rejoin target captured");
        assert!(json.contains("/v1/files"), "branch router route captured");
        // The flattened IRR-step router's routes, surfaced as a node field.
        assert!(json.contains("/v1/inference"), "step router route captured");
        // The IRR step-transition graph.
        assert!(
            json.contains("\"initial_step\": \"inference\""),
            "IRR entry step captured"
        );
        assert!(
            json.contains("\"next\": \"inference\""),
            "IRR transition target captured"
        );
        assert!(json.contains("\"value\": \"loop\""), "IRR transition trigger captured");
        // The carrier's `unless` condition.
        assert!(json.contains("unless"), "filter condition captured");
    }

    #[test]
    fn structure_detects_rejoin_drift() {
        let baseline = structure_json(TOPOLOGY_CONFIG);
        let drifted = structure_json(&TOPOLOGY_CONFIG.replace("rejoin: terminal", "rejoin: inference"));
        assert_ne!(
            baseline, drifted,
            "a changed branch rejoin target must alter the signature"
        );
    }

    #[test]
    fn structure_detects_condition_drift() {
        let baseline = structure_json(TOPOLOGY_CONFIG);
        let drifted = structure_json(&TOPOLOGY_CONFIG.replace("methods: [POST]", "methods: [GET]"));
        assert_ne!(baseline, drifted, "a changed filter condition must alter the signature");
    }

    #[test]
    fn structure_detects_branch_route_drift() {
        let baseline = structure_json(TOPOLOGY_CONFIG);
        let drifted =
            structure_json(&TOPOLOGY_CONFIG.replace("path_prefix: \"/v1/files\"", "path_prefix: \"/v1/embeddings\""));
        assert_ne!(baseline, drifted, "a changed branch route must alter the signature");
    }

    #[test]
    fn structure_detects_step_router_route_drift() {
        let baseline = structure_json(TOPOLOGY_CONFIG);
        let drifted = structure_json(&TOPOLOGY_CONFIG.replace("path: \"/v1/inference\"", "path: \"/v1/other\""));
        assert_ne!(
            baseline, drifted,
            "a changed IRR-step router route must alter the signature"
        );
    }

    #[test]
    fn structure_detects_irr_transition_drift() {
        let baseline = structure_json(TOPOLOGY_CONFIG);
        let drifted = structure_json(&TOPOLOGY_CONFIG.replace("value: loop", "value: continue"));
        assert_ne!(baseline, drifted, "a changed IRR transition must alter the signature");
    }
}
