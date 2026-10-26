// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! `cargo xtask sync-flow-visualizers` regenerates and verifies the checked-in
//! flow-visualizer HTML from the config it documents plus a curated sidecar.
//!
//! Each registered visualizer is rendered deterministically by
//! [`crate::flow_generator::render`], which merges the mechanically inferred
//! topology of the config (parsed through the pinned `praxis_core` via
//! [`crate::flow_graph::FlowGraph`]) with the curated narration in the sidecar,
//! cross-checking the two so structural drift fails loudly rather than rendering
//! silently. The command then either:
//!
//! * (default) compares the freshly rendered HTML byte-for-byte with the checked-in file and fails with an actionable
//!   diff when they differ, or
//! * (`--fix`) rewrites the checked-in HTML from the current config + sidecar.
//!
//! Because the HTML is generated, `make lint` catches any config change that
//! was not reflected in the visualizer, and any hand-edit of the generated HTML.

use std::path::{Path, PathBuf};

use clap::Parser;

use crate::{
    flow_generator::{self, KnobCheck, KnobSource},
    flow_graph::FlowGraph,
};

// -----------------------------------------------------------------------------
// Registry
// -----------------------------------------------------------------------------

/// One checked-in flow visualizer and the inputs it is generated from.
struct Visualizer {
    /// Source config, relative to the workspace root.
    config: &'static str,
    /// Static HTML template with `@@BLOCK@@` placeholders, relative to root.
    template: &'static str,
    /// Curated data sidecar (YAML), relative to root.
    sidecar: &'static str,
    /// Generated HTML output, relative to the workspace root.
    html: &'static str,
    /// The listener whose composed filter chains the HTML flattens and
    /// documents (the chains it references are concatenated in listener order).
    listener: &'static str,
    /// Scalar limits the HTML surfaces and the config source of each.
    knobs: &'static [KnobCheck],
}

/// Every checked-in flow visualizer managed by this command.
const VISUALIZERS: &[Visualizer] = &[Visualizer {
    config: "examples/configs/openai/responses/full-flow-agentic.yaml",
    template: "examples/configs/openai/responses/full-flow-agentic.visualizer.template.html",
    sidecar: "examples/configs/openai/responses/full-flow-agentic.visualizer.yaml",
    html: "examples/configs/openai/responses/full-flow-agentic.visualizer.html",
    listener: "ai-gateway",
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
        source: KnobSource::Filter("openai_file_search_dispatch"),
        key: "timeout_ms",
    },
    KnobCheck {
        knob: "file_search.max_response_bytes",
        source: KnobSource::Filter("openai_file_search_dispatch"),
        key: "max_response_bytes",
    },
    KnobCheck {
        knob: "file_search.max_total_response_bytes",
        source: KnobSource::Filter("openai_file_search_dispatch"),
        key: "max_total_response_bytes",
    },
    KnobCheck {
        knob: "file_search.max_state_bytes",
        source: KnobSource::Filter("openai_file_search_dispatch"),
        key: "max_state_bytes",
    },
    KnobCheck {
        knob: "file_search.on_failure",
        source: KnobSource::Filter("openai_file_search_dispatch"),
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
        source: KnobSource::Filter("openai_web_search_dispatch"),
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
    /// Rewrite the checked-in HTML from the current config + sidecar instead of
    /// only verifying it. Structural disagreements between the config and the
    /// sidecar are never auto-fixed; they always fail.
    #[arg(long)]
    fix: bool,
}

// -----------------------------------------------------------------------------
// Entry point
// -----------------------------------------------------------------------------

/// The result of processing one visualizer.
enum Report {
    /// The checked-in HTML already matches the freshly rendered output.
    UpToDate,
    /// The checked-in HTML was rewritten (`--fix`).
    Regenerated,
    /// The checked-in HTML is stale; carries an actionable first-difference.
    Stale(String),
    /// The config and sidecar disagree structurally; carries the differences.
    Errors(Vec<String>),
}

/// Regenerate or verify every checked-in flow visualizer.
///
/// Exits with status 1 when any visualizer is stale (without `--fix`) or when a
/// config/sidecar structural disagreement is found (in either mode), so
/// `make lint` fails.
pub(crate) fn run(args: &Args) {
    let root = workspace_root();
    let mut failed = false;
    for viz in VISUALIZERS {
        match process(&root, viz, args.fix) {
            Report::UpToDate => println!("{}: up to date", viz.html),
            Report::Regenerated => println!("{}: regenerated from {}", viz.html, viz.config),
            Report::Stale(diff) => {
                failed = true;
                eprintln!("{}: stale — run `cargo xtask sync-flow-visualizers --fix`", viz.html);
                eprintln!("{diff}");
            },
            Report::Errors(errors) => {
                failed = true;
                eprintln!(
                    "{}: {} structural difference(s) vs {} — edit the sidecar to match the config",
                    viz.sidecar,
                    errors.len(),
                    viz.config
                );
                for error in &errors {
                    eprintln!("  - {error}");
                }
            },
        }
    }
    if failed {
        std::process::exit(1);
    }
}

/// Render one visualizer and reconcile it with the checked-in HTML.
fn process(root: &Path, viz: &Visualizer, fix: bool) -> Report {
    let graph = match FlowGraph::from_file(&root.join(viz.config)) {
        Ok(graph) => graph,
        Err(err) => return Report::Errors(vec![err]),
    };
    let sidecar = match read_yaml(&root.join(viz.sidecar)) {
        Ok(value) => value,
        Err(errors) => return Report::Errors(errors),
    };
    let template = match std::fs::read_to_string(root.join(viz.template)) {
        Ok(text) => text,
        Err(err) => return Report::Errors(vec![format!("read {}: {err}", viz.template)]),
    };
    let generated = match flow_generator::render(&graph, viz.listener, &sidecar, &template, viz.knobs) {
        Ok(html) => html,
        Err(errors) => return Report::Errors(errors),
    };
    reconcile(root, viz, fix, &generated)
}

/// Compare the rendered HTML with the checked-in file, writing it under `--fix`.
fn reconcile(root: &Path, viz: &Visualizer, fix: bool, generated: &str) -> Report {
    let committed = std::fs::read_to_string(root.join(viz.html)).unwrap_or_default();
    if committed == generated {
        return Report::UpToDate;
    }
    if fix {
        return match std::fs::write(root.join(viz.html), generated) {
            Ok(()) => Report::Regenerated,
            Err(err) => Report::Errors(vec![format!("write {}: {err}", viz.html)]),
        };
    }
    Report::Stale(first_difference(&committed, generated))
}

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

/// Read and parse a YAML file into an order-preserving value.
fn read_yaml(path: &Path) -> Result<serde_yaml::Value, Vec<String>> {
    let text = std::fs::read_to_string(path).map_err(|err| vec![format!("read {}: {err}", path.display())])?;
    serde_yaml::from_str(&text).map_err(|err| vec![format!("parse {}: {err}", path.display())])
}

/// Describe the first line at which two texts diverge, for a stale-file report.
fn first_difference(committed: &str, generated: &str) -> String {
    for (index, (left, right)) in committed.lines().zip(generated.lines()).enumerate() {
        if left != right {
            return format!(
                "  first difference at line {}:\n    committed: {}\n    generated: {}",
                index + 1,
                truncate(left),
                truncate(right)
            );
        }
    }
    format!(
        "  files differ in length: committed {} lines, generated {} lines",
        committed.lines().count(),
        generated.lines().count()
    )
}

/// Truncate a line to a readable width on a character boundary.
fn truncate(line: &str) -> String {
    const MAX: usize = 120;
    if line.chars().count() <= MAX {
        return line.to_owned();
    }
    let mut short: String = line.chars().take(MAX).collect();
    short.push('…');
    short
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

    fn full_flow() -> &'static Visualizer {
        VISUALIZERS.first().expect("the full-flow visualizer is registered")
    }

    /// Describe a non-passing report for an assertion message.
    fn detail(report: &Report) -> String {
        match report {
            Report::UpToDate => "up to date".to_owned(),
            Report::Regenerated => "unexpectedly regenerated in check mode".to_owned(),
            Report::Stale(diff) => format!("stale:\n{diff}"),
            Report::Errors(errors) => format!("config/sidecar disagree: {errors:?}"),
        }
    }

    /// The checked-in HTML must be exactly what the generator produces from the
    /// committed config + sidecar; this is the regression gate wired into lint.
    #[test]
    fn full_flow_visualizer_is_up_to_date() {
        let root = workspace_root();
        let report = process(&root, full_flow(), false);
        assert!(
            matches!(report, Report::UpToDate),
            "run `cargo xtask sync-flow-visualizers --fix` — {}",
            detail(&report)
        );
    }

    /// Regeneration is deterministic: rendering twice yields identical bytes.
    #[test]
    fn regeneration_is_deterministic() {
        let root = workspace_root();
        let viz = full_flow();
        let graph = FlowGraph::from_file(&root.join(viz.config)).expect("config parses");
        let sidecar = read_yaml(&root.join(viz.sidecar)).expect("sidecar parses");
        let template = std::fs::read_to_string(root.join(viz.template)).expect("template readable");
        let first = flow_generator::render(&graph, viz.listener, &sidecar, &template, viz.knobs).expect("renders");
        let second = flow_generator::render(&graph, viz.listener, &sidecar, &template, viz.knobs).expect("renders");
        assert_eq!(first, second, "rendering must be deterministic");
    }

    #[test]
    fn truncate_keeps_short_lines_and_marks_long_ones() {
        assert_eq!(truncate("short"), "short");
        let long = "x".repeat(200);
        let cut = truncate(&long);
        assert!(cut.ends_with('…'));
        assert_eq!(cut.chars().count(), 121);
    }

    #[test]
    fn first_difference_reports_line_number() {
        let diff = first_difference("a\nb\nc", "a\nB\nc");
        assert!(diff.contains("line 2"), "got {diff}");
    }
}
