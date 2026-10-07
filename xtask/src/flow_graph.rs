// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Generic, faithful structural model of a Praxis configuration.
//!
//! [`FlowGraph`] is extracted from any config file through the pinned
//! `praxis_core` parser, so the topology it records — listeners, filter chains,
//! per-filter conditions and branch chains, and the opaque per-filter
//! configuration — is exactly what the proxy itself would load. Nothing here is
//! hand-authored: it is mechanically inferred structure only. Curated
//! behavioral narration lives in the visualizer sidecar, never in this model
//! (see [`crate::flow_visualizer`]).
//!
//! The model is consumed by the flow-visualizer tooling:
//!
//! * `sync-flow-visualizers` cross-checks the checked-in flow visualizer against its source config and fails CI on
//!   structural drift.
//! * `visualize-config` (a later increment) renders an arbitrary config, labelling unknown filters "semantics
//!   unavailable" rather than fabricating an explanation.

use std::path::Path;

use praxis_core::config::{Cluster, Config, FilterEntry, Listener};
use serde::Serialize;
use serde_json::{Map, Value as Json};

/// The filter type whose inference-step filters are hoisted inline when a chain
/// is flattened for display.
pub(crate) const IRR_FILTER: &str = "iterative_request_router";

/// Keys of a raw filter mapping that are modelled as first-class node fields
/// rather than opaque configuration.
const RESERVED_FILTER_KEYS: [&str; 6] = [
    "filter",
    "name",
    "conditions",
    "branch_chains",
    "response_conditions",
    "failure_mode",
];

// -----------------------------------------------------------------------------
// Model
// -----------------------------------------------------------------------------

/// The complete structural model of a parsed Praxis config.
#[derive(Debug, Serialize)]
pub(crate) struct FlowGraph {
    /// Every listener, in declaration order.
    pub(crate) listeners: Vec<FlowListener>,
    /// Every top-level named filter chain, in declaration order.
    pub(crate) filter_chains: Vec<FlowChain>,
    /// Every top-level cluster declaration, in declaration order.
    pub(crate) clusters: Vec<FlowCluster>,
    /// The global `insecure_options` block, rendered faithfully.
    pub(crate) insecure_options: Json,
    /// The global `runtime` block, rendered faithfully.
    pub(crate) runtime: Json,
}

/// One listener and the filter chains it references.
#[derive(Debug, Serialize)]
pub(crate) struct FlowListener {
    /// Listener name.
    pub(crate) name: String,
    /// Bind address (`host:port`).
    pub(crate) address: String,
    /// Wire protocol (defaulted when the YAML omits it).
    pub(crate) protocol: String,
    /// Names of the filter chains this listener serves.
    pub(crate) filter_chains: Vec<String>,
}

/// One named filter chain and its ordered filters.
#[derive(Debug, Serialize)]
pub(crate) struct FlowChain {
    /// Chain name.
    pub(crate) name: String,
    /// Filters in declaration order.
    pub(crate) filters: Vec<FlowFilter>,
}

/// One filter entry as declared in the YAML (no flattening applied).
#[derive(Debug, Clone, Serialize)]
pub(crate) struct FlowFilter {
    /// Filter type (the YAML `filter:` key).
    pub(crate) filter_type: String,
    /// Optional rejoin/display name.
    pub(crate) name: Option<String>,
    /// `when`/`unless` conditions as a JSON array (`[]` when none).
    pub(crate) conditions: Json,
    /// Branch chains as a JSON array (`[]` when none).
    pub(crate) branch_chains: Json,
    /// The opaque per-filter configuration (`{}` when none).
    pub(crate) config: Json,
}

/// One top-level cluster declaration.
#[derive(Debug, Serialize)]
pub(crate) struct FlowCluster {
    /// Cluster name.
    pub(crate) name: String,
    /// Endpoints as a JSON array.
    pub(crate) endpoints: Json,
    /// HTTP options, rendered faithfully.
    pub(crate) http: Json,
}

/// One node in a flattened chain view. Inference-step filters of an
/// `iterative_request_router` are hoisted inline after the router itself and
/// carry `depth = 1` and the owning `irr_step` name.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct FlowNode {
    /// 1-based position within the flattened chain.
    pub(crate) order: usize,
    /// Nesting depth (0 = main chain, 1 = inside an IRR step).
    pub(crate) depth: usize,
    /// Filter type (the YAML `filter:` key).
    pub(crate) filter_type: String,
    /// Optional rejoin/display name.
    pub(crate) name: Option<String>,
    /// `when`/`unless` conditions as a JSON array (`[]` when none).
    pub(crate) conditions: Json,
    /// Branch chains as a JSON array (`[]` when none).
    pub(crate) branch_chains: Json,
    /// The opaque per-filter configuration (`{}` when none).
    pub(crate) config: Json,
    /// The owning IRR step name when `depth == 1`, else `None`.
    pub(crate) irr_step: Option<String>,
}

// -----------------------------------------------------------------------------
// Extraction
// -----------------------------------------------------------------------------

impl FlowGraph {
    /// Parse a config from YAML text into the structural model.
    ///
    /// # Errors
    ///
    /// Returns the `praxis_core` parse/validation error as a string when the
    /// YAML is invalid.
    pub(crate) fn from_yaml_str(yaml: &str) -> Result<Self, String> {
        let config = Config::from_yaml(yaml).map_err(|err| err.to_string())?;
        Ok(Self::from_config(&config))
    }

    /// Parse a config file into the structural model.
    ///
    /// # Errors
    ///
    /// Returns a string error when the file cannot be read, parsed, or
    /// validated.
    pub(crate) fn from_file(path: &Path) -> Result<Self, String> {
        let yaml = std::fs::read_to_string(path).map_err(|err| format!("{}: {err}", path.display()))?;
        Self::from_yaml_str(&yaml).map_err(|err| format!("{}: {err}", path.display()))
    }

    /// Build the model from an already-parsed [`Config`].
    fn from_config(config: &Config) -> Self {
        Self {
            listeners: config.listeners.iter().map(flow_listener).collect(),
            filter_chains: config.filter_chains.iter().map(flow_chain).collect(),
            clusters: config.clusters.iter().map(flow_cluster).collect(),
            insecure_options: yaml_to_json(&to_yaml(&config.insecure_options)),
            runtime: yaml_to_json(&to_yaml(&config.runtime)),
        }
    }

    /// Flatten one filter chain into ordered nodes, hoisting the inference
    /// filters of any `iterative_request_router` step inline after it. Returns
    /// `None` when no chain of that name exists.
    pub(crate) fn flatten_chain(&self, chain_name: &str) -> Option<Vec<FlowNode>> {
        let chain = self.filter_chains.iter().find(|c| c.name == chain_name)?;
        let mut nodes = Vec::new();
        for filter in &chain.filters {
            push_filter_node(&mut nodes, filter);
        }
        Some(nodes)
    }
}

/// Build a [`FlowChain`] from a parsed filter chain.
fn flow_chain(chain: &praxis_core::config::FilterChainConfig) -> FlowChain {
    FlowChain {
        name: chain.name.clone(),
        filters: chain.filters.iter().map(flow_filter).collect(),
    }
}

/// Build a [`FlowFilter`] from a parsed filter entry.
fn flow_filter(entry: &FilterEntry) -> FlowFilter {
    FlowFilter {
        filter_type: entry.filter_type.clone(),
        name: entry.name.clone(),
        conditions: seq_to_json(&entry.conditions),
        branch_chains: opt_seq_to_json(entry.branch_chains.as_deref()),
        config: yaml_to_json(&entry.config),
    }
}

/// Build a [`FlowListener`] from a parsed listener.
fn flow_listener(listener: &Listener) -> FlowListener {
    FlowListener {
        name: listener.name.clone(),
        address: listener.address.clone(),
        protocol: yaml_scalar_string(&to_yaml(&listener.protocol)),
        filter_chains: listener.filter_chains.clone(),
    }
}

/// Build a [`FlowCluster`] from a parsed cluster.
fn flow_cluster(cluster: &Cluster) -> FlowCluster {
    FlowCluster {
        name: cluster.name.to_string(),
        endpoints: seq_to_json(&cluster.endpoints),
        http: yaml_to_json(&to_yaml(&cluster.http)),
    }
}

/// Push a filter's node, then hoist IRR step filters when applicable.
fn push_filter_node(nodes: &mut Vec<FlowNode>, filter: &FlowFilter) {
    let order = nodes.len() + 1;
    nodes.push(FlowNode::from_filter(order, filter));
    if filter.filter_type == IRR_FILTER {
        expand_irr_steps(nodes, &filter.config);
    }
}

/// Append one node per inference-step filter found in an IRR config.
fn expand_irr_steps(nodes: &mut Vec<FlowNode>, config: &Json) {
    let Some(steps) = config.get("steps").and_then(Json::as_array) else {
        return;
    };
    for step in steps {
        let step_name = step.get("name").and_then(Json::as_str).map(str::to_owned);
        let Some(filters) = step.get("filters").and_then(Json::as_array) else {
            continue;
        };
        for raw in filters {
            let order = nodes.len() + 1;
            nodes.push(FlowNode::from_raw(order, raw, step_name.clone()));
        }
    }
}

impl FlowNode {
    /// Build a depth-0 node from a declared filter.
    fn from_filter(order: usize, filter: &FlowFilter) -> Self {
        Self {
            order,
            depth: 0,
            filter_type: filter.filter_type.clone(),
            name: filter.name.clone(),
            conditions: filter.conditions.clone(),
            branch_chains: filter.branch_chains.clone(),
            config: filter.config.clone(),
            irr_step: None,
        }
    }

    /// Build a depth-1 node from a raw IRR step-filter mapping.
    fn from_raw(order: usize, raw: &Json, step: Option<String>) -> Self {
        let map = raw.as_object().cloned().unwrap_or_default();
        Self {
            order,
            depth: 1,
            filter_type: str_field(&map, "filter"),
            name: map.get("name").and_then(Json::as_str).map(str::to_owned),
            conditions: array_field(&map, "conditions"),
            branch_chains: array_field(&map, "branch_chains"),
            config: raw_config(&map),
            irr_step: step,
        }
    }
}

/// Read a string field from a JSON object, defaulting to empty.
fn str_field(map: &Map<String, Json>, key: &str) -> String {
    map.get(key).and_then(Json::as_str).unwrap_or_default().to_owned()
}

/// Read an array field from a JSON object, defaulting to `[]`.
fn array_field(map: &Map<String, Json>, key: &str) -> Json {
    map.get(key).cloned().unwrap_or_else(|| Json::Array(Vec::new()))
}

/// Collect every non-reserved key of a raw filter mapping into a config object.
fn raw_config(map: &Map<String, Json>) -> Json {
    let mut config = Map::new();
    for (key, value) in map {
        if !RESERVED_FILTER_KEYS.contains(&key.as_str()) {
            config.insert(key.clone(), value.clone());
        }
    }
    Json::Object(config)
}

// -----------------------------------------------------------------------------
// Serialization helpers
// -----------------------------------------------------------------------------

/// Serialize any value to a `serde_yaml::Value`, defaulting to null on error.
fn to_yaml<T: Serialize + ?Sized>(value: &T) -> serde_yaml::Value {
    serde_yaml::to_value(value).unwrap_or(serde_yaml::Value::Null)
}

/// Serialize a slice to a JSON array via YAML.
fn seq_to_json<T: Serialize>(items: &[T]) -> Json {
    yaml_to_json(&to_yaml(items))
}

/// Serialize an optional slice to a JSON array (`[]` when `None`).
fn opt_seq_to_json<T: Serialize>(items: Option<&[T]>) -> Json {
    items.map_or_else(|| Json::Array(Vec::new()), seq_to_json)
}

/// Convert a `serde_yaml::Value` into a `serde_json::Value` faithfully.
fn yaml_to_json(value: &serde_yaml::Value) -> Json {
    match value {
        serde_yaml::Value::Null => Json::Null,
        serde_yaml::Value::Bool(b) => Json::Bool(*b),
        serde_yaml::Value::Number(n) => yaml_number_to_json(n),
        serde_yaml::Value::String(s) => Json::String(s.clone()),
        serde_yaml::Value::Sequence(seq) => Json::Array(seq.iter().map(yaml_to_json).collect()),
        serde_yaml::Value::Mapping(map) => yaml_mapping_to_json(map),
        serde_yaml::Value::Tagged(tagged) => yaml_to_json(&tagged.value),
    }
}

/// Convert a YAML number into the closest JSON number.
fn yaml_number_to_json(number: &serde_yaml::Number) -> Json {
    if let Some(i) = number.as_i64() {
        Json::from(i)
    } else if let Some(u) = number.as_u64() {
        Json::from(u)
    } else if let Some(f) = number.as_f64() {
        serde_json::Number::from_f64(f).map_or(Json::Null, Json::Number)
    } else {
        Json::Null
    }
}

/// Convert a YAML mapping into a JSON object, stringifying non-string keys.
fn yaml_mapping_to_json(map: &serde_yaml::Mapping) -> Json {
    let mut out = Map::new();
    for (key, value) in map {
        out.insert(yaml_scalar_string(key), yaml_to_json(value));
    }
    Json::Object(out)
}

/// Render a YAML scalar (or fall back to compact YAML) as a plain string.
fn yaml_scalar_string(value: &serde_yaml::Value) -> String {
    match value {
        serde_yaml::Value::String(s) => s.clone(),
        serde_yaml::Value::Bool(b) => b.to_string(),
        serde_yaml::Value::Number(n) => n.to_string(),
        serde_yaml::Value::Null => "null".to_owned(),
        other => serde_yaml::to_string(other).unwrap_or_default().trim().to_owned(),
    }
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::too_many_lines,
    reason = "tests"
)]
mod tests {
    use super::*;

    /// The committed full-flow config, relative to the xtask crate root.
    const FULL_FLOW: &str = "../examples/configs/openai/responses/full-flow-agentic.yaml";

    /// The chain the visualizer flattens.
    const PIPELINE_CHAIN: &str = "full-flow-agentic-pipeline";

    /// A minimal valid config with `count` copies of `trace_context` followed
    /// by a `static_response`, for topology mutation tests.
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

    fn full_flow_graph() -> FlowGraph {
        FlowGraph::from_file(Path::new(FULL_FLOW)).expect("full-flow config should parse")
    }

    #[test]
    fn full_flow_flattens_to_thirty_one_ordered_nodes() {
        let graph = full_flow_graph();
        let nodes = graph.flatten_chain(PIPELINE_CHAIN).expect("chain exists");
        assert_eq!(nodes.len(), 31, "19 main + 12 IRR-step filters");
        for (i, node) in nodes.iter().enumerate() {
            assert_eq!(node.order, i + 1, "orders are 1-based and sequential");
        }
    }

    #[test]
    fn full_flow_node_types_and_depths_match_yaml() {
        let graph = full_flow_graph();
        let nodes = graph.flatten_chain(PIPELINE_CHAIN).expect("chain exists");
        assert_eq!(nodes[0].filter_type, "trace_context");
        assert_eq!(nodes[18].filter_type, IRR_FILTER, "IRR is the 19th filter");
        assert_eq!(nodes[18].depth, 0, "the IRR itself is main-chain");
        assert_eq!(nodes[19].depth, 1, "first inference-step filter is nested");
        assert_eq!(nodes[19].irr_step.as_deref(), Some("inference"));
        assert_eq!(nodes[19].filter_type, "project_state_owner_headers");
        assert_eq!(
            nodes[24].filter_type, "ai_guardrails",
            "guardrails precede the loop owner"
        );
        assert_eq!(nodes[25].filter_type, "openai_agentic_loop");
        assert_eq!(
            nodes[28].filter_type, "openai_responses_proxy",
            "proxy after the step load balancer"
        );
        assert_eq!(nodes[30].filter_type, "path_rewrite", "last step filter");
    }

    #[test]
    fn irr_router_node_retains_limits() {
        let graph = full_flow_graph();
        let nodes = graph.flatten_chain(PIPELINE_CHAIN).expect("chain exists");
        let router = nodes
            .iter()
            .find(|node| node.filter_type == IRR_FILTER)
            .expect("IRR router");
        assert_eq!(router.config.get("max_iterations").and_then(Json::as_u64), Some(8));
        assert_eq!(
            router.config.get("initial_step").and_then(Json::as_str),
            Some("inference")
        );
    }

    #[test]
    fn adding_a_filter_changes_node_count() {
        let two = FlowGraph::from_yaml_str(&minimal_config(&["trace_context", "trace_context"])).expect("valid config");
        let three = FlowGraph::from_yaml_str(&minimal_config(&["trace_context", "trace_context", "trace_context"]))
            .expect("valid config");
        assert_eq!(two.flatten_chain("c").expect("chain").len(), 2);
        assert_eq!(three.flatten_chain("c").expect("chain").len(), 3);
    }

    #[test]
    fn reordering_filters_changes_node_order() {
        let graph = FlowGraph::from_yaml_str(&minimal_config(&["state_owner", "trace_context"])).expect("valid config");
        let nodes = graph.flatten_chain("c").expect("chain");
        assert_eq!(nodes[0].filter_type, "state_owner");
        assert_eq!(nodes[1].filter_type, "trace_context");
    }

    #[test]
    fn removing_a_filter_changes_node_count_and_order() {
        let before = FlowGraph::from_yaml_str(&minimal_config(&["trace_context", "state_owner", "trace_context"]))
            .expect("valid config");
        let after =
            FlowGraph::from_yaml_str(&minimal_config(&["trace_context", "trace_context"])).expect("valid config");
        assert_eq!(before.flatten_chain("c").expect("chain").len(), 3);
        let after_nodes = after.flatten_chain("c").expect("chain");
        assert_eq!(after_nodes.len(), 2, "removing the middle filter drops one node");
        assert_eq!(
            after_nodes[1].filter_type, "trace_context",
            "the survivor shifts up into its slot"
        );
    }

    /// A config whose first filter carries a branch chain, exactly as the
    /// full-flow carrier does. Used to prove branch topology is captured
    /// verbatim on the node without being flattened into the main chain the way
    /// IRR inference steps are.
    const BRANCHED_CONFIG: &str = "listeners:\n  - name: l\n    address: 127.0.0.1:8080\n    filter_chains: [c]\nfilter_chains:\n  - name: c\n    filters:\n      - filter: headers\n        branch_chains:\n          - name: bypass\n            rejoin: terminal\n            chains:\n              - name: bypass-chain\n                filters:\n                  - filter: trace_context\n      - filter: state_owner\n";

    #[test]
    fn branch_chains_are_captured_without_being_flattened() {
        let graph = FlowGraph::from_yaml_str(BRANCHED_CONFIG).expect("branched config parses");
        let nodes = graph.flatten_chain("c").expect("chain");

        // Only the two declared main-chain filters are flattened; the branch's
        // inner trace_context is NOT hoisted inline (unlike IRR steps).
        assert_eq!(nodes.len(), 2, "branch chains are not flattened into the main chain");
        assert_eq!(nodes[0].filter_type, "headers");
        assert_eq!(nodes[1].filter_type, "state_owner");

        // ...but the full branch topology (name and inner filter) is recorded
        // verbatim on the carrier node, and the unbranched filter has none.
        let branches = nodes[0].branch_chains.as_array().expect("branch_chains is an array");
        assert_eq!(branches.len(), 1, "the single branch chain is recorded");
        let branch_text = serde_json::to_string(&nodes[0].branch_chains).expect("serializes");
        assert!(branch_text.contains("bypass"), "branch name preserved: {branch_text}");
        assert!(
            branch_text.contains("trace_context"),
            "branch inner filter preserved: {branch_text}"
        );
        assert!(
            nodes[1].branch_chains.as_array().is_some_and(Vec::is_empty),
            "the unbranched filter records no branch chains"
        );
    }
}
