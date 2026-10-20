#!/usr/bin/env python3
"""Publish a snapshot of exact-commit CI evidence alongside GPU qualification.

Uses only the standard library and gh. Job/step outcomes describe CI execution;
they do not imply per-test counts or complete provider/API conformance.
"""

import argparse
from collections import Counter
import json
from pathlib import Path

from vllm_qualification import command, compose_notes, now, read_json, render as render_gpu, write_json


START = "<!-- praxis:release-ci-qualification:start -->"
END = "<!-- praxis:release-ci-qualification:end -->"
# Keep the release inventory explicit so an absent workflow is visible too.
# Scope describes the workflow's purpose, not a claim that every case executed.
WORKFLOWS = (
    ("release.yaml", "Release test gate", "Unit, schema, and integration test steps"),
    ("tests.yaml", "Tests", "Rust unit tests, lint, feature matrix, and report checks"),
    ("integration.yaml", "Integration", "Schema, inference fixtures, integration, and SDK/client tests"),
    ("postgres.yaml", "PostgreSQL store", "PostgreSQL unit and integration tests"),
    ("valkey.yaml", "Valkey rate limits", "Valkey backend unit and integration tests"),
    ("vllm-integration.yaml", "vLLM integration", "Simulator, live CPU, and optional GPU/client jobs"),
    ("anthropic-vllm-vision.yaml", "Anthropic vision", "Anthropic SDK image requests on GPU"),
    ("openai-conformance.yaml", "OpenAI conformance", "Pinned schema and conformance checks"),
    ("openresponses-conformance.yaml", "OpenResponses conformance", "Responses translation against CPU vLLM"),
    ("container.yaml", "Container", "Container build and runtime smoke tests"),
    ("vllm-gpu-container.yaml", "vLLM GPU images", "Model image builds and smoke tests"),
    ("fips.yaml", "FIPS", "UBI image checks and FIPS host proof"),
    ("coverage.yaml", "Coverage", "Instrumented Rust tests"),
    ("msrv.yaml", "MSRV", "Minimum supported Rust compilation"),
    ("supply-chain.yaml", "Supply chain", "Dependency, license, and vulnerability checks"),
    ("codeql.yaml", "CodeQL", "Static security analysis"),
    ("documentation.yaml", "Documentation", "Rustdoc checks"),
    ("unicode-safety.yaml", "Unicode safety", "Hidden Unicode checks"),
    ("nightly.yaml", "Nightly", "Scheduled lint, documentation, tests, and audit"),
)
FAILURES = {"failure", "cancelled", "timed_out", "action_required", "startup_failure", "stale"}
# pull_request runs may check out a synthetic merge rather than head_sha.
EVENTS = {"push", "schedule", "workflow_dispatch", "merge_group"}


def api(endpoint):
    raw = command("gh", "api", endpoint)
    return json.loads(raw) if raw else None


def pages(endpoint, key):
    """Never return a partial successful inventory after a later page fails."""
    items = []
    for page in range(1, 11):  # Actions caps filtered run searches at 1,000.
        separator = "&" if "?" in endpoint else "?"
        data = api(f"{endpoint}{separator}per_page=100&page={page}")
        if not isinstance(data, dict) or not isinstance(data.get(key), list):
            raise ValueError(f"GitHub {key} lookup failed")
        batch = data[key]
        items.extend(batch)
        if len(items) >= data.get("total_count", float("inf")) or len(batch) < 100:
            return items
    raise ValueError(f"GitHub {key} pagination limit reached")


def status(record):
    return record.get("conclusion") or record.get("status") or "unavailable"


def eligible(run, repo, sha, workflow):
    return (run.get("head_sha") == sha
            and (run.get("head_repository") or {}).get("full_name") == repo
            and run.get("event") in EVENTS
            and run.get("path") == f".github/workflows/{workflow}")


def workflow_evidence(repo, sha, workflow, label, scope, release_run=None, attempt=None):
    entry = {"workflow": workflow, "name": label, "scope": scope,
             "status": "unavailable", "reason": None, "run": None, "jobs": []}
    try:
        if workflow == "release.yaml" and release_run is not None:
            run = api(f"repos/{repo}/actions/runs/{release_run}/attempts/{attempt}")
            if not isinstance(run, dict) or run.get("id") != release_run or run.get("run_attempt") != attempt:
                raise ValueError("release run or attempt mismatch")
        else:
            runs = pages(f"repos/{repo}/actions/workflows/{workflow}/runs?head_sha={sha}", "workflow_runs")
            matching = [run for run in runs if eligible(run, repo, sha, workflow)]
            if not matching:
                entry.update(status="not_run", reason="No eligible workflow run for the release commit")
                return entry
            run = max(matching, key=lambda item: item["id"])
        if not eligible(run, repo, sha, workflow):
            raise ValueError("workflow, repository, event, or release commit mismatch")
        entry["run"] = {key: run.get(key) for key in (
            "id", "run_attempt", "head_sha", "event", "status", "conclusion", "html_url", "created_at", "updated_at")}
        jobs = pages(f"repos/{repo}/actions/runs/{run['id']}/attempts/{run['run_attempt']}/jobs", "jobs")
        if any(job.get("run_id") != run["id"] or job.get("head_sha") != sha
               or job.get("run_attempt") != run["run_attempt"] for job in jobs):
            raise ValueError("job run, attempt, or release commit mismatch")
        if workflow == "release.yaml":
            # Publication is in progress when this snapshot is produced. Only
            # the completed test gate is evidence here, not the publishing job.
            jobs = [job for job in jobs if job.get("name") == "test"]
        entry["jobs"] = [
            {key: job.get(key) for key in (
                "id", "name", "status", "conclusion", "html_url", "started_at", "completed_at", "steps")}
            for job in jobs
        ]
        if not jobs:
            raise ValueError("No jobs recorded in the selected attempt")
        outcomes = [status(job) for job in jobs]
        entry["job_totals"] = dict(Counter(outcomes))
        entry["status"] = status(jobs[0]) if workflow == "release.yaml" else status(run)
        if entry["status"] == "success":
            if any(outcome in FAILURES for outcome in outcomes):
                entry["status"] = "failure"
                entry["reason"] = "Workflow succeeded but a job recorded failure"
            elif any(job.get("status") != "completed" for job in jobs) or any(
                    outcome not in ("success", "skipped", "neutral") for outcome in outcomes):
                entry["status"] = "incomplete"
                entry["reason"] = "Workflow succeeded but jobs are not complete"
            elif all(outcome in ("skipped", "neutral") for outcome in outcomes):
                entry["status"] = "skipped"
                entry["reason"] = "All jobs skipped or neutral; no execution evidence"
        if entry["reason"] is None:
            entry["reason"] = "Job and step outcomes from the selected attempt"
    except (ValueError, KeyError, TypeError) as error:
        entry.update(status="unavailable", reason=str(error))
    return entry


def collect(repo, sha, gpu, release_run=None, attempt=None):
    return {
        "schema_version": 1, "profile": "release-ci-evidence", "repository": repo,
        "release_sha": sha, "created_at": now(),
        "selection": "Newest eligible run per workflow for the exact release commit; latest attempt only. The current release run is pinned when supplied. No older green fallback.",
        "limitations": [
            "CI records workflow head SHA and job/step outcomes, not individual test counts or complete API conformance.",
            "PR runs are excluded because their checkout can be a synthetic merge; no evidence is borrowed from other commits.",
            "Missing runs may reflect path filters or optional triggers; no execution is inferred.",
            "This is a publication-time snapshot; queued or running jobs may finish later. Rerun Release to refresh it.",
            "GPU qualification identifies its tested checkout and binary separately from CI and the published container.",
        ],
        "workflows": [workflow_evidence(repo, sha, *item, release_run, attempt) for item in WORKFLOWS],
        "gpu": gpu,
    }


def cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ").replace("<", "&lt;").replace(">", "&gt;")


def render(report, detailed=False):
    lines = [START, "### Release CI evidence", "",
             f"Release commit: `{report['release_sha']}`. Snapshot: {report['created_at']}.", "",
             "Results below record CI jobs and steps. A successful workflow can include skipped tests; job counts are not test counts.", "",
             "| CI suite | Scope | Result | Jobs / missing evidence | Run |",
             "| --- | --- | --- | --- | --- |"]
    for entry in report["workflows"]:
        run = entry["run"]
        link = f"[run {run['id']}, attempt {run['run_attempt']}]({run['html_url']})" if run else "Unavailable"
        detail = ", ".join(f"{value} {key}" for key, value in sorted(entry.get("job_totals", {}).items()))
        if entry["status"] == "unavailable" or not detail:
            detail = entry["reason"]
        lines.append("| " + " | ".join(cell(value) for value in (
            entry["name"], entry["scope"], entry["status"], detail, link)) + " |")
    lines += ["", "See attached `release-qualification.md` for job and step outcomes, `release-qualification.json` for machine-readable CI and GPU evidence, and `qualification.json` for the original GPU report.",
              "", "Selection: " + report["selection"], ""]
    lines.extend(f"- {limitation}" for limitation in report["limitations"])
    gpu = report["gpu"]
    if gpu.get("status") in ("unavailable", "not_requested"):
        lines += ["", f"GPU evidence: **{gpu['status']}**. {cell(gpu.get('reason', 'No GPU report'))}.",
                  "To collect GPU evidence, dispatch `vllm-integration.yaml` with `run_live_vllm=true` on this release tag, wait for its summary, then rerun Release on the same tag."]
    if detailed:
        lines += ["", "#### CI job and step outcomes"]
        for entry in report["workflows"]:
            lines += ["", f"##### {entry['name']}", "", cell(entry["reason"])]
            for job in entry["jobs"]:
                lines += ["", f"- [{cell(job['name'])}]({job['html_url']}): **{status(job)}**"]
                for step in job.get("steps") or []:
                    lines.append(f"  - {cell(step['name'])}: {status(step)}")
    lines += [END, ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("repo", "sha", "gpu-report", "existing-notes", "output-dir"):
        parser.add_argument("--" + field, required=True)
    parser.add_argument("--run-id", type=int)
    parser.add_argument("--attempt", type=int)
    args = parser.parse_args()
    if (args.run_id is None) != (args.attempt is None):
        parser.error("--run-id and --attempt must be supplied together")
    gpu = read_json(args.gpu_report)
    if not isinstance(gpu, dict) or (
            gpu.get("workflow", {}).get("run_sha") or gpu.get("gateway", {}).get("checkout_sha")) != args.sha:
        parser.error("GPU report must identify the release commit")
    report = collect(args.repo, args.sha, gpu, args.run_id, args.attempt)
    output = Path(args.output_dir)
    write_json(output / "release-qualification.json", report)
    (output / "release-qualification.md").write_text(render(report, detailed=True) + "\n" + render_gpu(gpu, detailed=True), encoding="utf-8")
    notes = compose_notes(Path(args.existing_notes).read_text(encoding="utf-8"), render(report), START, END)
    (output / "release-notes.md").write_text(compose_notes(notes, render_gpu(gpu)), encoding="utf-8")


if __name__ == "__main__":
    main()
