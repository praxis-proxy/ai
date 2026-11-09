#!/usr/bin/env python3
"""Assemble the per-release support report from committed and release-time evidence.

The report answers one question: what did this release demonstrate, on which
commit, with which artifacts. It joins four sources that already exist and are
independently validated:

- `docs/conformance/translation-matrix.json`: declared native and translation
  behavior, generated from the drift-gated manifest.
- `docs/conformance/openai-conformance-report.json` and
  `docs/conformance/openresponses-coverage.json`: contract conformance and
  translation-oracle coverage.
- `qualification.json`: live GPU qualification, including per-scenario client
  acceptance results, when a run exists for the release commit.
- `release-qualification.json`: exact-commit CI workflow evidence.

Missing live evidence is rendered as `not validated`, never omitted, so a
release cannot quietly publish a claim it did not test. The script uses only
the standard library and runs on a hosted runner.

Scope is deliberately narrower than a product support matrix: it covers the
Praxis AI gateway and the vLLM backend the qualification suite uses. Model
gateway, API server, per-field API support, and live hosted-provider validation
are out of scope and named as such in the output.
"""

import argparse
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

VERSION = 1
PROFILE = "praxis-ai-release-support"
START = "<!-- praxis:support-report:start -->"
END = "<!-- praxis:support-report:end -->"

TRANSLATION_STATUSES = {
    "native", "passthrough", "translated", "translated-lossy",
    "rejected", "dropped", "n-a",
}

# What this report does and does not cover. Kept in one place so the rendered
# document and the JSON agree.
INCLUDES = [
    "Praxis AI gateway behavior in this repository",
    "Native vLLM inference backend exercised by qualification runs",
]
EXCLUDES = [
    "Model gateway (llm-d) and API server (OGX) layers",
    "Per-field Responses and Messages support for each layer",
    "Live validation of hosted providers (Azure OpenAI, AWS Bedrock, Google Vertex AI)",
    "Performance and latency benchmarking",
]

HARNESS_SUITES = (
    ("claude_acceptance", "Claude Code"),
    ("codex_acceptance", "Codex"),
    ("opencode_acceptance", "OpenCode"),
    ("pi_acceptance", "pi"),
)

HARNESS_PINS = (
    ("codex-cli", "Codex", "OpenAI Responses"),
    ("claude-code-cli", "Claude Code", "Anthropic Messages"),
    ("opencode-cli", "OpenCode 1.x", "OpenAI Chat Completions"),
    ("opencode-cli-v2", "OpenCode 2.x", "OpenAI Chat Completions"),
    ("pi-cli", "pi", "OpenAI Chat Completions"),
)

TEST_SUITES = (
    ("native_responses", "Native Responses"),
    ("translation", "Responses-to-Chat translation"),
    ("supporting", "Supporting"),
    ("credentialed_tools", "Credentialed tools"),
    ("unclassified", "Unclassified"),
)


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------

def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    """Return parsed JSON, or None when a path is absent or invalid."""
    if not path:
        return None
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def write(path, text):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text, encoding="utf-8")


def one_line(value):
    """Collapse arbitrary text onto one line for a table cell or bullet."""
    return " ".join(str("" if value is None else value).split())


def md_cell(value):
    return one_line(value).replace("|", "\\|").replace("<", "&lt;")


def counts(cells):
    result = {status: 0 for status in TRANSLATION_STATUSES}
    for cell in cells:
        status = cell.get("status")
        result[status] = result.get(status, 0) + 1
    return result


def table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(md_cell(item) for item in row) + " |")
    return lines


# -----------------------------------------------------------------------------
# Sections
# -----------------------------------------------------------------------------

def harness_pins(root):
    """Read the pinned client versions from the manifests CI already enforces."""
    rows = []
    for directory, client, api in HARNESS_PINS:
        manifest = Path(root) / "tests/integration/fixtures" / directory / "pin.toml"
        if not manifest.is_file():
            continue
        version = None
        for line in manifest.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("version") and "=" in stripped:
                version = stripped.split("=", 1)[1].strip().strip('"')
                break
        rows.append({"client": client, "api": api, "version": version,
                     "manifest": str(manifest.relative_to(root))})
    return rows


def translation_section(matrix):
    surfaces = matrix.get("surfaces") or []
    summary = []
    for surface in surfaces:
        totals = counts(surface.get("cells") or [])
        summary.append({
            "id": surface.get("id"),
            "short": surface.get("short"),
            "client_api": surface.get("client_api"),
            "backend_api": surface.get("backend_api"),
            "counts": totals,
        })
    return {
        "source": matrix.get("source"),
        "caveat": matrix.get("conformance_claim_caveat"),
        "preconditions": matrix.get("preconditions") or [],
        "status_totals": matrix.get("status_totals") or {},
        "surfaces": surfaces,
        "summary": summary,
        "cross_surface_gaps": matrix.get("cross_surface_gaps") or [],
    }


def conformance_section(openai, openresponses):
    section = {"openai_operations": None, "openresponses_translation": None}
    if isinstance(openai, dict):
        section["openai_operations"] = {
            "scope": openai.get("scope"),
            "reference_revision": (openai.get("reference") or {}).get("upstream", {}).get("revision"),
            "capability_coverage": openai.get("capability_coverage"),
            "runtime_verification": openai.get("runtime_verification"),
        }
    if isinstance(openresponses, dict):
        section["openresponses_translation"] = {
            "suite": openresponses.get("suite"),
            "coverage": openresponses.get("coverage"),
            "unsupported": openresponses.get("unsupported") or [],
            "inapplicable": openresponses.get("inapplicable") or [],
        }
    return section


def live_validation(qualification, sha):
    if not isinstance(qualification, dict):
        return {"status": "not validated",
                "reason": "No qualification report was produced for this release commit."}
    gateway = qualification.get("gateway") or {}
    tested = gateway.get("checkout_sha")
    if tested and sha and tested != sha:
        return {"status": "provenance mismatch",
                "reason": "Qualification tested checkout %s, not release commit %s" % (tested, sha),
                "workflow": qualification.get("workflow"),
                "gateway": gateway}
    return {
        "status": qualification.get("status", "unknown"),
        "reason": qualification.get("reason"),
        "requested": qualification.get("requested"),
        "workflow": qualification.get("workflow"),
        "gateway": gateway,
        "backend": qualification.get("backend") or {},
        "model": qualification.get("model") or {},
        "configuration": qualification.get("configuration") or {},
        "suites": qualification.get("suites") or {},
        "limitations": qualification.get("limitations") or [],
        "diagnostics": qualification.get("diagnostics") or [],
    }


def live_suite_rows(live):
    rows = []
    suites = live.get("suites") or {}
    for name, label in TEST_SUITES:
        suite = suites.get(name)
        if not isinstance(suite, dict):
            continue
        totals = suite.get("totals") or {}
        executed = sum(value for key, value in totals.items() if key != "unexecuted")
        rows.append({"suite": label, "name": name, "status": suite.get("status"),
                     "totals": totals, "executed": executed,
                     "reason": suite.get("reason")})
    return rows


def harness_rows(live):
    rows = []
    suites = live.get("suites") or {}
    for name, client in HARNESS_SUITES:
        suite = suites.get(name)
        if not isinstance(suite, dict):
            continue
        for case in suite.get("cases") or []:
            rows.append({"client": client, "suite": name,
                         "scenario": case.get("id"), "status": case.get("status")})
    return rows


def ci_evidence(release_qualification, sha):
    if not isinstance(release_qualification, dict):
        return {"status": "unavailable",
                "reason": "No exact-commit CI evidence was produced for this release.",
                "workflows": []}
    workflows = []
    for workflow in release_qualification.get("workflows") or []:
        run = workflow.get("run") or {}
        workflows.append({
            "name": workflow.get("name"),
            "workflow": workflow.get("workflow"),
            "scope": workflow.get("scope"),
            "status": workflow.get("status"),
            "reason": workflow.get("reason"),
            "job_totals": workflow.get("job_totals") or {},
            "url": run.get("html_url"),
        })
    return {
        "status": "recorded" if workflows else "unavailable",
        "release_sha": release_qualification.get("release_sha"),
        "selection": release_qualification.get("selection"),
        "limitations": release_qualification.get("limitations") or [],
        "workflows": workflows,
    }


def known_gaps(translation, live):
    gaps = []
    for gap in translation.get("cross_surface_gaps") or []:
        gaps.append({"source": "translation", "id": gap.get("id"),
                     "note": gap.get("summary"), "blocker": gap.get("blocker")})
    for surface in translation.get("surfaces") or []:
        for cell in surface.get("cells") or []:
            if cell.get("status") != "dropped":
                continue
            gaps.append({
                "source": "translation",
                "id": "%s/%s" % (surface.get("short"), cell.get("feature")),
                "note": cell.get("note"),
                "blocker": cell.get("blocker"),
            })
    for limitation in live.get("limitations") or []:
        gaps.append({"source": "qualification", "id": None,
                     "note": limitation.get("text"), "blocker": limitation.get("url")})
    return gaps


def cell_status_map(report):
    result = {}
    for surface in (report.get("translation") or {}).get("surfaces") or []:
        for cell in surface.get("cells") or []:
            result[(surface.get("id"), cell.get("feature"))] = cell.get("status")
    return result


def previous_diff(previous, current):
    if not isinstance(previous, dict):
        return None
    before = cell_status_map(previous)
    after = cell_status_map(current)
    if not before or not after:
        return None
    changes = []
    for key in sorted(set(before) | set(after)):
        old, new = before.get(key), after.get(key)
        if old != new:
            changes.append({"surface": key[0], "feature": key[1],
                            "from": old, "to": new})
    return {
        "release": (previous.get("release") or {}).get("tag"),
        "cells": changes,
    }


# -----------------------------------------------------------------------------
# Assembly
# -----------------------------------------------------------------------------

def build(args):
    root = Path(args.root)
    matrix = read_json(args.translation_matrix)
    openai = read_json(args.openai_conformance)
    openresponses = read_json(args.openresponses)
    qualification = read_json(args.qualification)
    release_qualification = read_json(args.release_qualification)
    previous = read_json(args.previous)

    missing = [name for name, value in (
        ("translation matrix", matrix),
        ("OpenAI conformance report", openai),
        ("OpenResponses coverage", openresponses),
    ) if not isinstance(value, dict)]
    if missing:
        raise SystemExit("support report inputs missing or invalid: " + ", ".join(missing))

    translation = translation_section(matrix)
    live = live_validation(qualification, args.sha)
    report = {
        "schema_version": VERSION,
        "profile": PROFILE,
        "release": {
            "tag": args.tag,
            "version": args.version or (args.tag or "").lstrip("v"),
            "commit": args.sha,
            "generated_at": args.generated_at or now(),
        },
        "scope": {"includes": INCLUDES, "excludes": EXCLUDES},
        "provenance": {
            "gateway": live.get("gateway") or {},
            "backend": live.get("backend") or {},
            "model": live.get("model") or {},
            "configuration": live.get("configuration") or {},
            "harness_pins": harness_pins(root),
        },
        "native_surfaces": matrix.get("native_surfaces") or [],
        "translation": translation,
        "conformance": conformance_section(openai, openresponses),
        "live_validation": live,
        "ci_evidence": ci_evidence(release_qualification, args.sha),
    }
    report["known_gaps"] = known_gaps(translation, live)
    report["previous"] = previous_diff(previous, report)
    report["evidence"] = "complete" if (
        report["live_validation"].get("status") in
        {"passed", "failed", "incomplete", "needs_review", "skipped", "unexecuted"}
        and report["ci_evidence"].get("workflows")
    ) else "partial"
    return report


# -----------------------------------------------------------------------------
# Rendering
# -----------------------------------------------------------------------------

def render_provenance(report, lines):
    provenance = report.get("provenance") or {}
    gateway = provenance.get("gateway") or {}
    backend = provenance.get("backend") or {}
    model = provenance.get("model") or {}
    configuration = provenance.get("configuration") or {}
    dependencies = ((report.get("live_validation") or {}).get("suites", {})
                    .get("native_responses") or {}).get("dependencies") or {}

    rows = [
        ("Release", report["release"].get("tag")),
        ("Commit", report["release"].get("commit")),
        ("Praxis core", gateway.get("praxis_version")),
        ("Gateway binary SHA-256", gateway.get("binary_sha256")),
        ("vLLM", backend.get("version")),
        ("vLLM image", backend.get("local_image_id")),
        ("Model", model.get("identifier")),
        ("Model revision", model.get("resolved_revision")),
        ("Reference config", configuration.get("reference_path")),
        ("Config SHA-256", configuration.get("sha256")),
        ("OpenAI SDK", dependencies.get("openai")),
    ]
    lines += ["### Provenance", ""]
    lines += table(("Item", "Value"), [(label, value or "unavailable") for label, value in rows])
    lines.append("")


def render_scope(report, lines):
    scope = report.get("scope") or {}
    lines += ["### Scope", ""]
    lines.append("This report covers:")
    lines.append("")
    for item in scope.get("includes") or []:
        lines.append("- " + item)
    lines += ["", "It does not cover:", ""]
    for item in scope.get("excludes") or []:
        lines.append("- " + item)
    lines.append("")


def render_native(report, lines):
    surfaces = report.get("native_surfaces") or []
    lines += ["### Native API surface", ""]
    if not surfaces:
        lines += ["No native surfaces are declared.", ""]
        return
    rows = [(surface.get("client_api"), surface.get("operations") or "—",
             surface.get("status"), surface.get("note"))
            for surface in surfaces]
    lines += table(("Surface", "Operations", "Status", "Notes"), rows)
    lines.append("")


def render_translation(report, lines):
    translation = report.get("translation") or {}
    lines += ["### Translation support", ""]
    summary = translation.get("summary") or []
    if not summary:
        lines += ["No translation surfaces are declared.", ""]
        return
    rows = []
    for surface in summary:
        totals = surface.get("counts") or {}
        rows.append((
            surface.get("short"),
            "%s -> %s" % (surface.get("client_api"), surface.get("backend_api")),
            totals.get("translated", 0),
            totals.get("translated-lossy", 0),
            totals.get("passthrough", 0),
            totals.get("rejected", 0),
            totals.get("dropped", 0),
            totals.get("n-a", 0),
        ))
    lines += table(("Surface", "Client to backend", "Translated", "Lossy",
                    "Passthrough", "Rejected", "Gaps", "n/a"), rows)
    lines.append("")
    totals = translation.get("status_totals") or {}
    lines.append("Cells: " + ", ".join(
        "%s %s" % (key, totals[key]) for key in sorted(totals)) + ".")
    lines.append("")
    lines.append("A gap is a known defect, not a support level. Every gap carries "
                 "a blocker in `support-report.json` and in the translation matrix.")
    lines.append("")


def render_conformance(report, lines):
    conformance = report.get("conformance") or {}
    lines += ["### Contract conformance", ""]

    openai = conformance.get("openai_operations")
    if not openai:
        lines += ["**OpenAI operations.** Unavailable for this release.", ""]
    else:
        coverage = openai.get("capability_coverage") or {}
        verification = openai.get("runtime_verification") or {}
        passed = {True: "passed", False: "failed"}.get(
            verification.get("all_passed"), "unknown")
        lines += [
            "**OpenAI operations.** %s/%s operations covered (%s%%); "
            "runtime verification %s." % (
                coverage.get("covered_operations_count", "?"),
                coverage.get("total_operations", "?"),
                coverage.get("coverage_percent", "?"),
                passed),
            "",
        ]
        rows = [(area.get("area"), area.get("total"), area.get("covered"),
                 area.get("coverage_percent")) for area in coverage.get("by_area") or []]
        if rows:
            lines += table(("Area", "Operations", "Covered", "Percent"), rows)
            lines.append("")

    openresponses = conformance.get("openresponses_translation")
    if not openresponses:
        lines += ["**OpenResponses translation.** Unavailable for this release.", ""]
    else:
        coverage = openresponses.get("coverage") or {}
        lines += [
            "**OpenResponses translation.** %s/%s in-scope templates passing "
            "(%s%%); %s of %s suite templates excluded as inapplicable." % (
                coverage.get("supported", "?"),
                coverage.get("in_scope", "?"),
                coverage.get("percent", "?"),
                coverage.get("inapplicable", "?"),
                coverage.get("total", "?")),
            "",
        ]


def render_live(report, lines):
    live = report.get("live_validation") or {}
    lines += ["### Live validation", ""]
    status = live.get("status", "not validated")
    lines.append("Status: **%s**. %s" % (status, one_line(live.get("reason"))))
    lines.append("")
    if status in ("not validated", "provenance mismatch", "unavailable", "not_requested"):
        lines.append("No live claims are made for this release commit.")
        lines.append("")
        return

    workflow = live.get("workflow") or {}
    if workflow.get("url"):
        lines += ["[Qualification run](%s) (attempt %s)." % (
            workflow.get("url"), workflow.get("attempt")), ""]

    suite_rows = live_suite_rows(live)
    if suite_rows:
        lines.append("Suites:")
        lines.append("")
        lines += table(("Suite", "Status", "Executed cases", "Reason"),
                       [(row["suite"], row["status"], row["executed"], row["reason"])
                        for row in suite_rows])
        lines.append("")

    harness = harness_rows(live)
    if harness:
        lines.append("Client acceptance scenarios:")
        lines.append("")
        lines += table(("Client", "Scenario", "Status"),
                       [(row["client"], row["scenario"], row["status"]) for row in harness])
        lines.append("")


def render_pins(report, lines):
    pins = (report.get("provenance") or {}).get("harness_pins") or []
    if not pins:
        return
    lines += ["### Pinned clients", ""]
    lines += table(("Client", "API", "Version", "Manifest"),
                   [(pin.get("client"), pin.get("api"), pin.get("version") or "unavailable",
                     "`%s`" % pin.get("manifest")) for pin in pins])
    lines.append("")


def render_ci(report, lines):
    evidence = report.get("ci_evidence") or {}
    lines += ["### Exact-commit CI evidence", ""]
    workflows = evidence.get("workflows") or []
    if not workflows:
        lines += ["No exact-commit CI evidence was recorded for this release.", ""]
        return
    rows = []
    for workflow in workflows:
        totals = workflow.get("job_totals") or {}
        jobs = ", ".join("%s %s" % (key, totals[key]) for key in sorted(totals)) or "—"
        rows.append((workflow.get("name"), workflow.get("status"), jobs,
                     workflow.get("url") or "—"))
    lines += table(("Workflow", "Status", "Jobs", "Run"), rows)
    lines.append("")


def render_gaps(report, lines):
    gaps = report.get("known_gaps") or []
    lines += ["### Known gaps and limitations", ""]
    if not gaps:
        lines += ["None recorded.", ""]
        return
    for gap in gaps:
        note = one_line(gap.get("note")) or "no description"
        blocker = gap.get("blocker")
        reference = " (%s)" % one_line(blocker) if blocker else ""
        lines.append("- %s: %s%s" % (gap.get("id") or gap.get("source"), note, reference))
    lines.append("")


def render_diff(report, lines):
    previous = report.get("previous")
    if not previous:
        return
    lines += ["### Changes since %s" % (previous.get("release") or "the previous release"), ""]
    changes = previous.get("cells") or []
    if not changes:
        lines += ["No translation cell changed status.", ""]
        return
    lines += table(("Surface", "Feature", "Was", "Now"),
                   [(change.get("surface"), change.get("feature"),
                     change.get("from"), change.get("to")) for change in changes])
    lines.append("")


def render(report):
    lines = [START, "## Release support report", ""]
    release = report.get("release") or {}
    lines.append("`%s` at `%s`; generated %s. Evidence: **%s**." % (
        release.get("tag") or "unreleased",
        release.get("commit") or "unknown",
        release.get("generated_at"),
        report.get("evidence", "partial")))
    lines.append("")
    lines.append("This report is assembled from the committed conformance reports "
                 "and the exact-commit qualification evidence for this release. "
                 "Missing evidence is stated, not omitted.")
    lines.append("")
    render_scope(report, lines)
    render_provenance(report, lines)
    render_native(report, lines)
    render_translation(report, lines)
    render_conformance(report, lines)
    render_live(report, lines)
    render_pins(report, lines)
    render_ci(report, lines)
    render_gaps(report, lines)
    render_diff(report, lines)
    lines.append("The complete machine-readable report is `support-report.json`.")
    lines.append(END)
    lines.append("")
    return "\n".join(lines)


def compose_notes(existing, section, start=START, end=END):
    """Replace this report's section in release notes, preserving the rest."""
    text = existing or ""
    block = "%s\n%s\n%s" % (start, section.strip(), end)
    if start in text and end in text:
        before = text.split(start, 1)[0]
        after = text.split(end, 1)[1]
        return before + block + after
    return text.rstrip() + "\n\n" + block + "\n"


# -----------------------------------------------------------------------------
# Verification
# -----------------------------------------------------------------------------

def verify(report, markdown, tag=None, sha=None):
    problems = []
    if not isinstance(report, dict):
        return ["support report is not a JSON object"]
    if report.get("schema_version") != VERSION or report.get("profile") != PROFILE:
        problems.append("schema version or profile mismatch")
    release = report.get("release") or {}
    if tag and release.get("tag") != tag:
        problems.append("release tag mismatch: %r" % release.get("tag"))
    if sha and release.get("commit") != sha:
        problems.append("release commit mismatch: %r" % release.get("commit"))
    translation = report.get("translation") or {}
    surfaces = translation.get("surfaces") or []
    if not surfaces:
        problems.append("no translation surfaces recorded")
    for surface in surfaces:
        for cell in surface.get("cells") or []:
            if cell.get("status") not in TRANSLATION_STATUSES:
                problems.append("invalid translation status %r in %s/%s" % (
                    cell.get("status"), surface.get("id"), cell.get("feature")))
    if not report.get("native_surfaces"):
        problems.append("no native surfaces recorded")
    if not (report.get("conformance") or {}).get("openai_operations"):
        problems.append("OpenAI conformance section is missing")
    if not isinstance(report.get("ci_evidence"), dict):
        problems.append("exact-commit CI evidence section is missing")
    if not isinstance(report.get("live_validation"), dict):
        problems.append("live validation section is missing")
    if report.get("evidence") not in ("complete", "partial"):
        problems.append("evidence state is missing or invalid")
    if not markdown or START not in markdown or END not in markdown:
        problems.append("markdown is missing the support report markers")
    elif "## Release support report" not in markdown:
        problems.append("markdown is missing the report heading")
    return problems


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def generate(args):
    report = build(args)
    markdown = render(report)
    output = Path(args.output_dir)
    write(output / "support-report.json", json.dumps(report, indent=2) + "\n")
    write(output / "support-report.md", markdown)

    if args.existing_notes:
        try:
            existing = Path(args.existing_notes).read_text(encoding="utf-8")
        except OSError:
            existing = ""
        notes = compose_notes(existing, markdown.replace(START, "").replace(END, ""))
        write(output / "release-notes.md", notes)

    problems = verify(report, markdown, args.tag, args.sha)
    if problems:
        for problem in problems:
            print("support report error: " + problem, file=sys.stderr)
        raise SystemExit(1)

    live = report["live_validation"]["status"]
    print("support report: %s; %d translation surfaces; %d known gaps; live validation %s" % (
        report["release"]["tag"], len(report["translation"]["surfaces"]),
        len(report["known_gaps"]), live))


def verify_command(args):
    report = read_json(args.report)
    try:
        markdown = Path(args.markdown).read_text(encoding="utf-8")
    except OSError:
        markdown = ""
    problems = verify(report, markdown, args.tag, args.sha)
    if problems:
        for problem in problems:
            print("support report error: " + problem, file=sys.stderr)
        raise SystemExit(1)
    print("support report is valid")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="repository root")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser("generate", help="assemble the release report")
    generate_parser.add_argument("--tag", required=True)
    generate_parser.add_argument("--sha", required=True)
    generate_parser.add_argument("--version", default=None)
    generate_parser.add_argument("--generated-at", default=None)
    generate_parser.add_argument("--translation-matrix",
                                 default="docs/conformance/translation-matrix.json")
    generate_parser.add_argument("--openai-conformance",
                                 default="docs/conformance/openai-conformance-report.json")
    generate_parser.add_argument("--openresponses",
                                 default="docs/conformance/openresponses-coverage.json")
    generate_parser.add_argument("--qualification", default=None)
    generate_parser.add_argument("--release-qualification", default=None)
    generate_parser.add_argument("--previous", default=None,
                                 help="previous support-report.json for a release diff")
    generate_parser.add_argument("--existing-notes", default=None)
    generate_parser.add_argument("--output-dir", required=True)
    generate_parser.set_defaults(func=generate)

    verify_parser = subparsers.add_parser("verify", help="validate a generated report")
    verify_parser.add_argument("--report", required=True)
    verify_parser.add_argument("--markdown", required=True)
    verify_parser.add_argument("--tag", default=None)
    verify_parser.add_argument("--sha", default=None)
    verify_parser.set_defaults(func=verify_command)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
