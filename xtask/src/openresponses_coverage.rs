// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! `cargo xtask openresponses-coverage` computes the `OpenResponses` translation
//! conformance coverage from the triage manifest and regenerates (or verifies)
//! `docs/conformance/openresponses-translation.md`.
//!
//! Coverage is `supported / (supported + unsupported)` — the fraction of
//! in-scope suite templates the `responses_to_chat_completions` filter passes.
//! `inapplicable` templates (wrong transport, a backend modality the pinned CPU
//! model lacks, operations outside the translation surface) are excluded from
//! the denominator. Fixing a translation gap promotes an `unsupported` template
//! to `supported` and bumps the number, so the report is a ratchet that tracks
//! implementation quality rather than a static all-pass claim.

use std::{fmt::Write as _, fs, path::PathBuf};

use clap::Parser;
use serde::Deserialize;

// -----------------------------------------------------------------------------
// CLI
// -----------------------------------------------------------------------------

/// CLI arguments for `cargo xtask openresponses-coverage`.
#[derive(Parser)]
pub(crate) struct Args {
    /// Write the generated report instead of checking for drift.
    #[arg(long)]
    fix: bool,
}

/// Triage manifest, relative to the workspace root.
const MANIFEST_REL: &str = "tests/conformance/openresponses/manifest.yaml";

/// Generated report, relative to the workspace root.
const DOC_REL: &str = "docs/conformance/openresponses-translation.md";

// -----------------------------------------------------------------------------
// Manifest
// -----------------------------------------------------------------------------

/// The subset of the triage manifest this tool reads.
#[derive(Deserialize)]
struct Manifest {
    /// Pinned external suite coordinates.
    suite: Suite,
    /// In-scope templates the filter passes today (numerator).
    supported: Vec<Entry>,
    /// In-scope templates not passing yet (denominator headroom).
    unsupported: Vec<Entry>,
    /// Out-of-scope templates, excluded from the denominator.
    inapplicable: Vec<Entry>,
}

/// Pinned coordinates of the external `OpenResponses` suite.
#[derive(Deserialize)]
struct Suite {
    /// Pinned suite commit SHA.
    commit: String,
    /// Pinned bun runner version.
    bun: String,
    /// Pinned zod version.
    zod: String,
}

/// One triaged suite template.
#[derive(Deserialize)]
struct Entry {
    /// Template id as enumerated by the suite.
    id: String,
}

// -----------------------------------------------------------------------------
// Entry Point
// -----------------------------------------------------------------------------

/// Verify or regenerate the coverage report from the manifest.
pub(crate) fn run(args: &Args) {
    let root = workspace_root();
    let manifest_path = root.join(MANIFEST_REL);
    let doc_path = root.join(DOC_REL);

    let raw = fs::read_to_string(&manifest_path).expect("failed to read OpenResponses triage manifest");
    let manifest: Manifest = serde_yaml::from_str(&raw).expect("failed to parse OpenResponses triage manifest");

    let content = render_doc(&manifest);

    if args.fix {
        fs::write(&doc_path, &content).unwrap();
        println!("wrote {}", doc_path.strip_prefix(&root).unwrap_or(&doc_path).display());
    } else {
        let current = fs::read_to_string(&doc_path).unwrap_or_default();
        if current == content {
            println!("{DOC_REL} is up to date");
        } else {
            eprintln!("{DOC_REL} is stale");
            eprintln!("\nrun: cargo xtask openresponses-coverage --fix");
            std::process::exit(1);
        }
    }
}

// -----------------------------------------------------------------------------
// Coverage
// -----------------------------------------------------------------------------

/// Coverage figures derived from the manifest bucket sizes.
struct Coverage {
    /// Number of supported (passing) templates.
    supported: usize,
    /// Number of unsupported (in-scope, not-yet-passing) templates.
    unsupported: usize,
    /// Number of inapplicable (out-of-scope) templates.
    inapplicable: usize,
}

impl Coverage {
    /// Templates the filter is responsible for (the denominator).
    fn in_scope(&self) -> usize {
        self.supported + self.unsupported
    }

    /// Every triaged template.
    fn total(&self) -> usize {
        self.in_scope() + self.inapplicable
    }

    /// Fraction of in-scope templates passing, as a percentage.
    #[expect(
        clippy::cast_precision_loss,
        reason = "template counts are tiny and far below f64 mantissa precision"
    )]
    fn percent(&self) -> f64 {
        let in_scope = self.in_scope();
        if in_scope == 0 {
            100.0
        } else {
            (self.supported as f64 / in_scope as f64) * 100.0
        }
    }
}

/// Derive [`Coverage`] from a manifest.
fn coverage(m: &Manifest) -> Coverage {
    Coverage {
        supported: m.supported.len(),
        unsupported: m.unsupported.len(),
        inapplicable: m.inapplicable.len(),
    }
}

/// Join a bucket's ids into a comma-separated list.
fn ids(entries: &[Entry]) -> String {
    entries.iter().map(|e| e.id.as_str()).collect::<Vec<_>>().join(", ")
}

// -----------------------------------------------------------------------------
// Rendering
// -----------------------------------------------------------------------------

/// Render the full `openresponses-translation.md` document.
#[expect(clippy::too_many_lines, reason = "straight-line Markdown document template")]
fn render_doc(m: &Manifest) -> String {
    let cov = coverage(m);
    let mut out = String::new();

    writeln!(out, "# OpenResponses Translation Conformance").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "<!-- Generated by `cargo xtask openresponses-coverage --fix`. Do not edit"
    )
    .unwrap();
    writeln!(
        out,
        "     by hand — edit tests/conformance/openresponses/manifest.yaml and regenerate. -->"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "This suite runs the external [OpenResponses][openresponses] conformance oracle"
    )
    .unwrap();
    writeln!(
        out,
        "against the `responses_to_chat_completions` translation filter — never against"
    )
    .unwrap();
    writeln!(out, "native Responses passthrough.").unwrap();
    writeln!(out).unwrap();

    writeln!(out, "## Coverage").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "**{supported} / {in_scope} in-scope templates passing ({percent:.1}%).** \
         {inapplicable} of {total}",
        supported = cov.supported,
        in_scope = cov.in_scope(),
        percent = cov.percent(),
        inapplicable = cov.inapplicable,
        total = cov.total(),
    )
    .unwrap();
    writeln!(
        out,
        "suite templates are out of scope and excluded from the denominator."
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "Coverage counts templates the translation filter is responsible for"
    )
    .unwrap();
    writeln!(
        out,
        "(`supported` + `unsupported`). Fixing a translation gap promotes an"
    )
    .unwrap();
    writeln!(out, "`unsupported` template to `supported` and bumps this number.").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "- **Supported ({}):** {}", cov.supported, ids(&m.supported)).unwrap();
    writeln!(out, "- **Unsupported ({}):** {}", cov.unsupported, ids(&m.unsupported)).unwrap();
    writeln!(
        out,
        "- **Inapplicable ({}, excluded):** {}",
        cov.inapplicable,
        ids(&m.inapplicable)
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "Per-template rationale lives in the triage manifest").unwrap();
    writeln!(
        out,
        "([`tests/conformance/openresponses/manifest.yaml`][manifest]); the launcher"
    )
    .unwrap();
    writeln!(out, "fails if any suite template is missing from it.").unwrap();
    writeln!(out).unwrap();

    writeln!(out, "## What runs").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "A Praxis listener loads the shared").unwrap();
    writeln!(
        out,
        "[`examples/configs/openai/responses/responses-to-chat-completions.yaml`][example]"
    )
    .unwrap();
    writeln!(
        out,
        "example, which translates `POST /v1/responses` into `POST /v1/chat/completions`"
    )
    .unwrap();
    writeln!(
        out,
        "and forwards to a Chat-Completions-only backend (vLLM CPU, `Qwen/Qwen3-0.6B`)."
    )
    .unwrap();
    writeln!(
        out,
        "A translation-witness shim sits between Praxis and vLLM and asserts the"
    )
    .unwrap();
    writeln!(
        out,
        "backend only ever receives `POST /v1/chat/completions` (criterion f)."
    )
    .unwrap();
    writeln!(out).unwrap();

    writeln!(out, "## Pins").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "- Suite commit: `{}`", m.suite.commit).unwrap();
    writeln!(out, "- Runner: bun `{}`, zod `{}`", m.suite.bun, m.suite.zod).unwrap();
    writeln!(out, "- Backend: vLLM CPU, `Qwen/Qwen3-0.6B`").unwrap();
    writeln!(out).unwrap();

    writeln!(out, "## Running locally").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "```console").unwrap();
    writeln!(out, "make test-responses-conformance").unwrap();
    writeln!(out, "```").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "Requires `bun`, `uv`, and a reachable vLLM backend (`VLLM_BASE_URL`, default"
    )
    .unwrap();
    writeln!(out, "`http://127.0.0.1:8000`).").unwrap();
    writeln!(out).unwrap();

    writeln!(out, "[openresponses]: https://github.com/openresponses/openresponses").unwrap();
    writeln!(out, "[manifest]: ../../tests/conformance/openresponses/manifest.yaml").unwrap();
    writeln!(
        out,
        "[example]: ../../examples/configs/openai/responses/responses-to-chat-completions.yaml"
    )
    .unwrap();

    out
}

// -----------------------------------------------------------------------------
// Utilities
// -----------------------------------------------------------------------------

/// Find the workspace root by looking for the top-level `Cargo.toml`.
fn workspace_root() -> PathBuf {
    let output = std::process::Command::new("cargo")
        .args(["locate-project", "--workspace", "--message-format=plain"])
        .output()
        .expect("failed to run cargo locate-project");
    let path = String::from_utf8(output.stdout).expect("non-utf8 path");
    PathBuf::from(path.trim())
        .parent()
        .expect("Cargo.toml has no parent")
        .to_owned()
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_manifest() -> Manifest {
        let yaml = "
suite:
  repo: https://example.invalid/openresponses
  commit: deadbeef
  bun: 1.3.12
  zod: 3.25.76
supported:
  - id: basic-response
  - id: system-prompt
  - id: multi-turn
  - id: assistant-phase
  - id: tool-calling
  - id: streaming-response
unsupported:
  - id: response-output-phase-schema
    reason: capability gap
inapplicable:
  - id: image-input
    reason: vision backend
  - id: websocket-response
    reason: transport
";
        serde_yaml::from_str(yaml).expect("sample manifest parses")
    }

    #[test]
    fn coverage_counts_only_in_scope() {
        let cov = coverage(&sample_manifest());
        assert_eq!(cov.supported, 6);
        assert_eq!(cov.unsupported, 1);
        assert_eq!(cov.inapplicable, 2);
        assert_eq!(cov.in_scope(), 7);
        assert_eq!(cov.total(), 9);
    }

    #[test]
    fn coverage_percent_is_supported_over_in_scope() {
        let cov = Coverage {
            supported: 6,
            unsupported: 1,
            inapplicable: 10,
        };
        assert!((cov.percent() - 85.714_285).abs() < 0.001, "got {}", cov.percent());
    }

    #[test]
    fn coverage_percent_all_supported_when_no_in_scope() {
        let cov = Coverage {
            supported: 0,
            unsupported: 0,
            inapplicable: 5,
        };
        assert!((cov.percent() - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn ids_join_preserves_manifest_order() {
        let m = sample_manifest();
        assert_eq!(
            ids(&m.supported),
            "basic-response, system-prompt, multi-turn, assistant-phase, tool-calling, streaming-response"
        );
    }

    #[test]
    fn render_is_deterministic() {
        let m = sample_manifest();
        assert_eq!(render_doc(&m), render_doc(&m));
    }

    #[test]
    fn render_reports_headline_number_and_pins() {
        let doc = render_doc(&sample_manifest());
        assert!(
            doc.contains("6 / 7 in-scope templates passing (85.7%)"),
            "headline missing:\n{doc}"
        );
        assert!(doc.contains("2 of 9"), "out-of-scope count missing:\n{doc}");
        assert!(
            doc.contains("response-output-phase-schema"),
            "unsupported id missing:\n{doc}"
        );
        assert!(doc.contains("Suite commit: `deadbeef`"), "commit pin missing:\n{doc}");
        assert!(
            doc.contains("bun `1.3.12`, zod `3.25.76`"),
            "runner pins missing:\n{doc}"
        );
    }
}
