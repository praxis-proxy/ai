#!/usr/bin/env python3
"""Capture and publish exact-commit vLLM Responses gateway evidence.

No third-party dependencies: this runs on both the GPU AMI and hosted runners.
The report describes the locally built gateway binary and model-baked image,
never the separately published release container.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from datetime import datetime, timezone

VERSION = 2
PROFILE = "responses-gateway-live-vllm-gpu"
ARTIFACT = "vllm-qualification"
# Keep the existing note delimiters so release reruns replace earlier sections.
START = "<!-- praxis:native-vllm-qualification:start -->"
END = "<!-- praxis:native-vllm-qualification:end -->"
OUTCOMES = ("passed", "failed", "skipped", "xfailed", "xpassed", "unexecuted")
LIMITATIONS = [
    {"text": "Selected text and tool behavior over native Responses and Responses-to-Chat HTTP/SSE paths; no complete OpenAI API conformance claim.", "url": None},
    {"text": "Streamed Conversation append remains outside the supported contract.", "url": "https://github.com/praxis-proxy/ai/issues/410"},
    {"text": "Background, WebSockets, and multimodal models are excluded.", "url": None},
    {"text": "Credentialed provider tests and optional client acceptance are separate evidence.", "url": None},
]


def now():
    return datetime.now(timezone.utc).isoformat()


def command(*argv):
    try:
        return subprocess.check_output(argv, text=True, stderr=subprocess.DEVNULL, timeout=30).strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def file_hash(path):
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def read_json(path):
    if not path:
        return None
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def locked_version(name):
    # The GPU runner's system Python is 3.9, so tomllib is not available.
    try:
        lock = Path("Cargo.lock").read_text(encoding="utf-8")
    except OSError:
        return None
    pattern = r'\[\[package\]\]\s+name = "' + re.escape(name) + r'"\s+version = "([^"]+)"'
    match = re.search(pattern, lock)
    return match.group(1) if match else None


def model_revision(runtime, model):
    if not model:
        return None
    # huggingface_hub's local-dir download metadata records the resolved commit
    # per file. Only report it when all available files agree on one SHA.
    code = (
        "from pathlib import Path; import re,sys; "
        "root=Path('/opt/vllm/models')/sys.argv[1]/'.cache/huggingface/download'; "
        "values={p.read_text(errors='replace').splitlines()[0] for p in root.rglob('*.metadata') if p.is_file()}; "
        "commits={v for v in values if re.fullmatch('[0-9a-f]{40}',v)}; "
        "print(next(iter(commits)) if len(commits)==1 and len(values)==1 else '')"
    )
    return command(runtime, "exec", "vllm", "python", "-c", code, model) or None


def suite(name, path, profile=None):
    result = read_json(path)
    if not result:
        return {"name": name, "status": "unexecuted", "reason": "pytest result file missing", "selected": [], "deselected": [], "dependencies": {}}
    all_cases = result.get("selected", [])
    cases = [case for case in all_cases if case.get("profile", "unclassified") == profile] if profile else all_cases
    deselected = result.get("deselected", [])
    if profile:
        deselected = [case for case in deselected if case.get("profile", "unclassified") == profile]
    totals = {outcome: sum(case.get("outcome") == outcome for case in cases) for outcome in OUTCOMES}
    executed = sum(totals[key] for key in OUTCOMES if key != "unexecuted")
    # A pytest exit of 1 from a failed case in another profile does not make
    # this profile incomplete. Other nonzero exits still mean the run broke.
    process_completed = result.get("exit_code") == 0 or (
        result.get("exit_code") == 1
        and any(case.get("outcome") in ("failed", "xpassed") for case in all_cases)
    )
    if not executed:
        status, reason = "incomplete", "zero selected cases executed"
    elif totals["failed"]:
        status, reason = "failed", "selected tests failed"
    elif totals["xpassed"]:
        status, reason = "needs_review", "unexpected passes need contract review"
    elif totals["unexecuted"] or not process_completed:
        status, reason = "incomplete", "test process did not complete cleanly"
    elif totals["skipped"] == executed:
        status, reason = "skipped", "all selected cases skipped; no provider proof"
    elif not totals["passed"]:
        status, reason = "incomplete", "no selected case passed"
    else:
        status, reason = "passed", "selected cases completed"
    return {"name": name, "status": status, "reason": reason, "selected": cases,
            "deselected": deselected, "totals": totals,
            "exit_code": result.get("exit_code"), "dependencies": result.get("dependencies", {}),
            "started_at": result.get("started_at"), "finished_at": result.get("finished_at")}


def capture(args):
    runtime = os.environ.get("CONTAINER_RUNTIME", "docker")
    image = os.environ.get("VLLM_BUILT_GPU_IMAGE")
    model = os.environ.get("VLLM_GPU_MODEL")
    run_id = os.environ.get("GITHUB_RUN_ID")
    repo = os.environ.get("GITHUB_REPOSITORY")
    config = "examples/configs/openai/responses/full-flow-agentic.yaml"
    native = suite("native_responses", args.responses, "native")
    translation = suite("translation", args.responses, "translation")
    supporting = suite("supporting", args.responses, "supporting")
    unclassified = suite("unclassified", args.responses, "unclassified")
    if not unclassified["selected"]:
        unclassified["status"] = "empty"
        unclassified["reason"] = "all selected cases attributed"
    credentialed = suite("credentialed_tools", args.credentialed)
    groups = (("native Responses", native), ("translation", translation),
              ("supporting", supporting), ("credentialed tools", credentialed))
    status, reason = "passed", "native and translation cases completed"
    if args.setup != "passed":
        status, reason = "incomplete", "GPU setup failed before qualification completed"
    else:
        problem = next(((label, group) for severity in ("failed", "needs_review")
                        for label, group in groups if group["status"] == severity), None)
        if problem:
            label, group = problem
            status, reason = group["status"], f"{label}: {group['reason']}"
        else:
            if unclassified["selected"]:
                status, reason = "incomplete", "selected cases lack a known gateway profile"
            else:
                for label, group in groups:
                    allowed = ("passed", "skipped") if label in ("supporting", "credentialed tools") else ("passed",)
                    if label == "supporting" and not group["selected"]:
                        continue
                    if group["status"] not in allowed:
                        status, reason = "incomplete", f"{label}: {group['reason']}"
                        break
    gpu = command("nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total", "--format=csv,noheader")
    image_id = command(runtime, "image", "inspect", image, "--format={{.Id}}") if image else None
    resolved_revision = model_revision(runtime, model)
    report = {
        "schema_version": VERSION,
        "profile": PROFILE,
        "requested": True,
        "created_at": now(),
        "workflow": {"run_id": int(run_id) if run_id else None,
                     "attempt": int(os.environ.get("GITHUB_RUN_ATTEMPT", "1")),
                     "url": f"https://github.com/{repo}/actions/runs/{run_id}" if repo and run_id else None,
                     "event": os.environ.get("GITHUB_EVENT_NAME")},
        "gateway": {"checkout_sha": command("git", "rev-parse", "HEAD"),
                    "binary": "target/debug/praxis-ai",
                    "binary_sha256": file_hash("target/debug/praxis-ai"),
                    "build_profile": "cargo build -p praxis-ai-proxy --features full (debug)",
                    "praxis_version": locked_version("praxis-proxy"),
                    "praxis_filter_version": locked_version("praxis-proxy-filter")},
        "backend": {"mode": "real-vllm-gpu", "version": command(runtime, "exec", "vllm", "python", "-c", "import vllm; print(vllm.__version__)"),
                    "image_tag": image, "local_image_id": image_id,
                    "registry_digest": None,
                    "containerfile_sha256": file_hash("vllm/Containerfile"),
                    "base_image": "docker.io/vllm/vllm-openai:v0.30.0",
                    "base_image_id": command(runtime, "image", "inspect", "docker.io/vllm/vllm-openai:v0.30.0", "--format={{.Id}}")},
        "model": {"identifier": model, "resolved_revision": resolved_revision,
                  "revision_note": None if resolved_revision else "HF snapshot revision unavailable from local build metadata"},
        "configuration": {"reference_path": config, "sha256": file_hash(config),
                          "additional_config_sha256": {
                              path: file_hash(path) for path in (
                                  "examples/configs/openai/responses/agentic-loop.yaml",
                                  "examples/configs/openai/responses/irr-terminal-streaming.yaml",
                                  "examples/configs/openai/responses/responses-to-chat-completions.yaml",
                                  "examples/configs/openai/responses/responses-to-chat-completions-reasoning.yaml",
                                  "examples/configs/openai/responses/compact.yaml",
                                  "examples/configs/openai/responses/web-search-chat-completions.yaml",
                                  "examples/configs/openai/responses/client-tool-compat.yaml",
                                  "examples/configs/openai/responses/client-tool-compat-chat-completions.yaml",
                              )},
                          "effective_overrides": {"VLLM_MODEL": model, "VLLM_TEST_BACKEND": "live", "DATABASE_URL": "PostgreSQL (credential redacted)"},
                          "ogx_constraints_sha256": file_hash("tests/integration/ogx-constraints.txt"),
                          "storage_backend": "postgresql"},
        "runner": {"name": os.environ.get("RUNNER_NAME"), "os": os.environ.get("RUNNER_OS"),
                   "architecture": os.environ.get("RUNNER_ARCH"), "gpu": gpu,
                   "instance_type": os.environ.get("GPU_INSTANCE_TYPE")},
        "suites": {"native_responses": native, "translation": translation,
                   "supporting": supporting, "unclassified": unclassified,
                   "credentialed_tools": credentialed,
                   "simulator": {"status": "separate_job", "reason": "separate hosted simulator jobs"}},
        "phases": {"provisioning": "passed", "setup": args.setup,
                   "test_execution": status},
        "status": status,
        "reason": reason,
        "limitations": LIMITATIONS,
        "diagnostics": ["vllm-gpu-responses-raw artifact: pytest output (including focused HTTP/SSE failure assertions), case JSON, and available service logs"],
    }
    write_json(args.output, report)


def acceptance(path, job_result, run_id, attempt, sha, scope):
    raw = read_json(path)
    if (not isinstance(raw, dict) or raw.get("run_id") != run_id
            or raw.get("attempt") != attempt or raw.get("checkout_sha") != sha
            or not isinstance(raw.get("cases"), list)):
        return {"status": "unreported" if job_result == "success" else job_result,
                "job_status": job_result, "scope": scope,
                "reason": "per-scenario result artifact unavailable or provenance mismatch",
                "cases": []}
    return {"status": job_result, "job_status": job_result, "scope": scope,
            "cases": raw["cases"], "reason": "scenario statuses from individual GitHub steps"}


def passing_suite_problem(report):
    suites = report.get("suites", {})
    for name, profile in (("native_responses", "native"), ("translation", "translation")):
        group = suites.get(name, {})
        cases = group.get("selected", [])
        if (group.get("status") != "passed" or not cases
                or not any(case.get("outcome") == "passed" for case in cases)
                or any(case.get("profile") != profile
                       or case.get("outcome") not in ("passed", "skipped", "xfailed")
                       for case in cases)):
            return f"passing report lacks executed {profile} cases with valid attribution"
    supporting = suites.get("supporting", {})
    if supporting.get("selected") and (
        supporting.get("status") not in ("passed", "skipped")
        or any(case.get("profile") != "supporting"
               or case.get("outcome") not in ("passed", "skipped", "xfailed")
               for case in supporting["selected"])
    ):
        return "passing report has invalid supporting case results"
    if suites.get("unclassified", {}).get("selected"):
        return "passing report contains unclassified selected cases"
    if suites.get("credentialed_tools", {}).get("status") not in ("passed", "skipped"):
        return "passing report lacks completed credentialed tool results"
    return None


def finish(raw, requested, start_result, suite_result, claude_result, codex_result, run_id, attempt, url, sha,
           claude_raw=None, codex_raw=None, opencode_result="skipped", opencode_raw=None):
    report = read_json(raw) if raw else None
    raw_valid = (isinstance(report, dict) and report.get("schema_version") == VERSION
                 and report.get("profile") == PROFILE
                 and all(isinstance(report.get(key), dict) for key in
                         ("workflow", "gateway", "backend", "model", "configuration", "suites"))
                 and report.get("workflow", {}).get("run_id") == run_id
                 and report.get("workflow", {}).get("attempt") == attempt)
    if not raw_valid:
        report = {"schema_version": VERSION, "profile": PROFILE, "requested": requested,
                  "created_at": now(), "gateway": {"checkout_sha": None}, "backend": {},
                  "model": {}, "configuration": {}, "runner": {}, "suites": {},
                  "limitations": LIMITATIONS, "diagnostics": []}
        report["status"] = "incomplete" if requested else "not_requested"
        report["reason"] = ("GPU provisioning failed or no current-attempt Responses result artifact was produced"
                            if requested else "Responses GPU qualification was not requested")
    report["workflow"] = {"run_id": run_id, "attempt": attempt, "url": url}
    report["requested"] = requested
    report["phases"] = {"provisioning": start_result,
                        "setup": report.get("phases", {}).get("setup", "unknown"),
                        "test_execution": suite_result}
    report["suites"]["claude_acceptance"] = acceptance(
        claude_raw, claude_result, run_id, attempt, sha,
        "native Messages, translated Chat, Anthropic SDK and optional tools")
    report["suites"]["codex_acceptance"] = acceptance(
        codex_raw, codex_result, run_id, attempt, sha,
        "native Responses and translated Chat")
    report["suites"]["opencode_acceptance"] = acceptance(
        opencode_raw, opencode_result, run_id, attempt, sha,
        "native Chat Completions text turn and tool-call round trip")
    if not requested:
        report["status"], report["reason"] = "not_requested", "Responses GPU qualification was not requested"
    elif suite_result != "success" and report["status"] == "passed":
        report["status"] = "incomplete"
        report["reason"] = f"selected cases passed but GPU suite job concluded {suite_result}"
    if requested and report["status"] == "passed":
        problem = passing_suite_problem(report)
        if problem:
            report["status"], report["reason"] = "incomplete", problem
    if requested and report["status"] == "passed" and not report.get("gateway", {}).get("checkout_sha"):
        report["status"], report["reason"] = "incomplete", "actual checkout SHA unavailable"
    if requested and report.get("gateway", {}).get("checkout_sha") and report["gateway"]["checkout_sha"] != sha:
        report["status"], report["reason"] = "incomplete", "actual checkout SHA differs from workflow SHA"
    if requested and report["status"] == "passed":
        required = {
            "gateway binary hash": report.get("gateway", {}).get("binary_sha256"),
            "Praxis core version": report.get("gateway", {}).get("praxis_version"),
            "vLLM version": report.get("backend", {}).get("version"),
            "local vLLM image ID": report.get("backend", {}).get("local_image_id"),
            "model ID": report.get("model", {}).get("identifier"),
            "reference config hash": report.get("configuration", {}).get("sha256"),
            "OpenAI SDK version": report.get("suites", {}).get("native_responses", {}).get("dependencies", {}).get("openai"),
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            report["status"], report["reason"] = "incomplete", "required provenance unavailable: " + ", ".join(missing)
    report["workflow"]["run_sha"] = sha
    return report


def totals(report, name):
    return report.get("suites", {}).get(name, {}).get("totals", {})


def render(report, detailed=False):
    status = report.get("status", "unavailable")
    lines = [START, "### vLLM Responses gateway qualification", ""]
    if status == "unavailable":
        lines.append("vLLM Responses gateway qualification: unavailable for this release commit. No usable exact-commit GPU qualification report was found.")
    elif status == "not_requested":
        lines.append("vLLM Responses gateway qualification: GPU testing was not requested.")
    else:
        lines.append(f"vLLM Responses gateway qualification: **{status.replace('_', ' ')}**. {report.get('reason', '')}")
    gateway, backend, model = (report.get(key, {}) for key in ("gateway", "backend", "model"))
    if status not in ("unavailable", "not_requested"):
        lines += ["", f"Tested checkout: `{gateway.get('checkout_sha') or 'unavailable'}`; locally built gateway binary `{gateway.get('binary_sha256') or 'unavailable'}` (debug/full); Praxis core `{gateway.get('praxis_version') or 'unavailable'}`.",
                  f"Backend: vLLM `{backend.get('version') or 'unavailable'}`, local image `{backend.get('local_image_id') or 'unavailable'}`; model `{model.get('identifier') or 'unavailable'}` (revision `{model.get('resolved_revision') or 'unavailable'}`)."]
        config = report.get("configuration", {})
        deps = report.get("suites", {}).get("native_responses", {}).get("dependencies", {})
        lines.append(f"SDK: OpenAI Python `{deps.get('openai') or 'unavailable'}`; config `{config.get('reference_path') or 'unavailable'}` (`{config.get('sha256') or 'unavailable'}`); storage `{config.get('storage_backend') or 'unavailable'}`.")
        for name, label in (("native_responses", "Native Responses"),
                            ("translation", "Responses-to-Chat translation"),
                            ("supporting", "Supporting")):
            group = report.get("suites", {}).get(name, {})
            counts = totals(report, name)
            lines.append(f"{label}: {group.get('status', 'unavailable')}; selected cases: "
                         + ", ".join(f"{key} {counts.get(key, 0)}" for key in OUTCOMES) + ".")
        if report.get("suites", {}).get("unclassified", {}).get("selected"):
            lines.append(f"Unclassified selected cases: {len(report['suites']['unclassified']['selected'])} (profile attribution required).")
        for name in ("credentialed_tools", "claude_acceptance", "codex_acceptance", "opencode_acceptance"):
            suite_data = report.get("suites", {}).get(name, {})
            cases = suite_data.get("cases", [])
            suffix = f" ({sum(case.get('status') == 'success' for case in cases)}/{len(cases)} scenarios succeeded)" if cases else ""
            lines.append(f"{name.replace('_', ' ').title()}: {suite_data.get('status', 'unavailable')}{suffix}.")
    if report.get("workflow", {}).get("url"):
        lines += ["", f"[Workflow run]({report['workflow']['url']}) and [report/diagnostic artifacts]({report['workflow']['url']}#artifacts)"]
    if detailed:
        lines += ["", "Expected failures document known contract boundaries; unexpected passes require review of the supported contract.",
                  "", "Known limitations:"]
        for limitation in report.get("limitations", []):
            label = limitation["text"]
            lines.append(f"- [{label}]({limitation['url']})" if limitation.get("url") else f"- {label}")
        for suite_name in ("native_responses", "translation", "supporting", "unclassified", "credentialed_tools"):
            suite_data = report.get("suites", {}).get(suite_name, {})
            notable = [case for case in suite_data.get("selected", []) if case.get("outcome") != "passed"]
            if notable:
                lines += ["", f"{suite_name.replace('_', ' ').title()} case reasons (complete list in `qualification.json`):"]
                for case in notable[:20]:
                    reason = (str(case.get("reason", "")).splitlines() or [""])[0][:240]
                    sdk_info = f" [SDK {case['sdk_version']}]" if case.get("sdk_version") else f" [SDK {case['sdk_lane']}]" if case.get("sdk_lane") else ""
                    lines.append(f"- `{case['id']}`{sdk_info}: {case.get('outcome')} | {reason}")
                if len(notable) > 20:
                    lines.append(f"- {len(notable) - 20} further cases in the artifact")
        for suite_name in ("claude_acceptance", "codex_acceptance", "opencode_acceptance"):
            cases = report.get("suites", {}).get(suite_name, {}).get("cases", [])
            if cases:
                lines += ["", f"{suite_name.replace('_', ' ').title()} scenarios:"]
                for case in cases:
                    lines.append(f"- `{case['id']}`: {case['status']}")
                    pytest_cases = case.get("pytest", {}).get("selected", [])
                    for item in pytest_cases:
                        if item.get("outcome") != "passed":
                            reason = (str(item.get("reason", "")).splitlines() or [""])[0][:160]
                            lines.append(f"  - `{item['id']}`: {item.get('outcome')} | {reason}")
    else:
        lines += ["", "Limitations: selected text and tool behavior over native Responses and Responses-to-Chat HTTP/SSE paths only; [streamed Conversation append](https://github.com/praxis-proxy/ai/issues/410) and multimodal/background/WebSocket paths are excluded. See the attached `qualification.json` for case reasons and separate client/tool results."]
    lines += [END, ""]
    return "\n".join(lines)


def compose_notes(existing, section):
    if START in existing and END in existing:
        return existing[:existing.index(START)] + section.rstrip() + existing[existing.index(END) + len(END):]
    return existing.rstrip() + "\n\n" + section


def unavailable(sha, reason, url=None):
    return {"schema_version": VERSION, "profile": PROFILE, "requested": True,
            "created_at": now(), "status": "unavailable", "reason": reason,
            "gateway": {"checkout_sha": None}, "workflow": {"run_sha": sha, "url": url},
            "backend": {}, "model": {}, "configuration": {}, "suites": {},
            "limitations": LIMITATIONS}


def choose_run(runs, jobs_by_id):
    """Newest completed Responses GPU attempt; never fall back after invalid evidence."""
    for run in sorted(runs, key=lambda item: item["id"], reverse=True):
        if run.get("status") != "completed":
            continue
        jobs = jobs_by_id.get(run["id"], [])
        native = next((job for job in jobs if job.get("name") == "Run complete live vLLM Responses suite (GPU)"), None)
        start = next((job for job in jobs if job.get("name") == "Start ephemeral EC2 GPU runner"), None)
        if native and native.get("conclusion") != "skipped":
            return run
        if start and start.get("conclusion") in ("failure", "cancelled", "timed_out", "action_required"):
            return run
        if start and start.get("conclusion") == "success" and native and native.get("conclusion") == "skipped":
            acceptance = [job for job in jobs if job.get("name") in (
                "Claude Code and Anthropic SDK acceptance on real vLLM (GPU)",
                "Codex acceptance on real vLLM (GPU)")]
            if acceptance and all(job.get("conclusion") == "skipped" for job in acceptance):
                return run
    return None


def verify(report, run, sha):
    if not isinstance(report, dict):
        return "report missing or invalid JSON"
    if report.get("schema_version") != VERSION or report.get("profile") != PROFILE:
        return "report version or Responses gateway profile mismatch"
    if not all(isinstance(report.get(key), dict) for key in ("workflow", "gateway", "backend", "model", "configuration", "suites")):
        return "report structure invalid"
    if report.get("workflow", {}).get("run_id") != run["id"] or report.get("workflow", {}).get("attempt") != run["run_attempt"]:
        return "report run or attempt mismatch"
    if report["workflow"].get("run_sha") not in (None, sha):
        return "report workflow SHA mismatch"
    if run.get("head_sha") != sha or report.get("gateway", {}).get("checkout_sha") != sha:
        return "workflow or actual checkout SHA mismatch"
    if not report.get("requested"):
        return "Responses GPU testing was not requested"
    if report.get("status") not in ("passed", "failed", "incomplete", "needs_review", "skipped", "unexecuted"):
        return "report status invalid"
    if report.get("status") == "passed":
        problem = passing_suite_problem(report)
        if problem:
            return problem
        native = report["suites"]["native_responses"]
        if not native.get("dependencies", {}).get("openai"):
            return "passing report lacks OpenAI SDK version"
        for section, field in (("gateway", "binary_sha256"), ("gateway", "praxis_version"),
                               ("backend", "version"), ("backend", "local_image_id"),
                               ("model", "identifier"), ("configuration", "sha256")):
            if not report[section].get(field):
                return f"passing report lacks {section}.{field}"
    return None


def gh_json(endpoint):
    result = command("gh", "api", endpoint)
    try:
        return json.loads(result) if result else None
    except ValueError:
        return None


def release_evidence(args):
    sha = args.sha
    repo = args.repo
    runs = []
    page = 1
    while True:
        runs_data = gh_json(f"repos/{repo}/actions/workflows/vllm-integration.yaml/runs?head_sha={sha}&status=completed&per_page=100&page={page}")
        if not isinstance(runs_data, dict):
            break
        batch = runs_data.get("workflow_runs", [])
        runs.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    if not isinstance(runs_data, dict):
        report = unavailable(sha, "workflow run lookup failed")
    else:
        jobs_by_id = {}
        for run in runs:
            data = gh_json(f"repos/{repo}/actions/runs/{run['id']}/jobs?per_page=100&filter=latest")
            if not isinstance(data, dict):
                report = unavailable(sha, f"jobs lookup failed for run {run['id']}")
                break
            jobs_by_id[run["id"]] = data.get("jobs", [])
        else:
            chosen = choose_run(runs, jobs_by_id)
            if chosen is None:
                report = unavailable(sha, "no completed Responses GPU attempt for this commit")
            else:
                run_id = chosen["id"]
                url = chosen.get("html_url")
                destination = Path(args.output_dir)
                destination.mkdir(parents=True, exist_ok=True)
                try:
                    downloaded = subprocess.run(["gh", "run", "download", str(run_id), "--repo", repo,
                                                 "--name", ARTIFACT, "--dir", str(destination / "download")],
                                                capture_output=True, text=True, check=False, timeout=120)
                except (OSError, subprocess.TimeoutExpired):
                    downloaded = None
                if downloaded is None or downloaded.returncode != 0:
                    report = unavailable(sha, "report artifact missing, expired, or download failed", url)
                else:
                    raw = read_json(destination / "download" / "qualification.json")
                    problem = verify(raw, chosen, sha)
                    report = unavailable(sha, problem, url) if problem else raw
    output = Path(args.output_dir)
    write_json(output / "qualification.json", report)
    (output / "qualification-section.md").write_text(render(report), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    cap = sub.add_parser("capture")
    cap.add_argument("--responses", required=True)
    cap.add_argument("--credentialed", required=True)
    cap.add_argument("--setup", choices=("passed", "failed"), required=True)
    cap.add_argument("--output", required=True)
    fin = sub.add_parser("finish")
    fin.add_argument("--raw", required=True)
    fin.add_argument("--requested", choices=("true", "false"), required=True)
    for field in ("start-result", "suite-result", "claude-result", "codex-result", "run-id", "attempt", "url", "sha", "output", "markdown", "claude-raw", "codex-raw"):
        fin.add_argument("--" + field, required=True)
    # Optional so a dispatch that predates the OpenCode lane still parses.
    fin.add_argument("--opencode-result", default="skipped")
    fin.add_argument("--opencode-raw", default=None)
    rel = sub.add_parser("release-evidence")
    for field in ("repo", "sha", "output-dir"):
        rel.add_argument("--" + field, required=True)
    notes = sub.add_parser("compose")
    for field in ("existing", "section", "output"):
        notes.add_argument("--" + field, required=True)
    args = parser.parse_args()
    if args.command == "capture":
        capture(args)
    elif args.command == "finish":
        report = finish(args.raw, args.requested == "true", args.start_result, args.suite_result,
                        args.claude_result, args.codex_result, int(args.run_id), int(args.attempt), args.url, args.sha,
                        args.claude_raw, args.codex_raw, args.opencode_result, args.opencode_raw)
        write_json(args.output, report)
        Path(args.markdown).write_text(render(report, detailed=True), encoding="utf-8")
    elif args.command == "release-evidence":
        release_evidence(args)
    elif args.command == "compose":
        Path(args.output).write_text(compose_notes(Path(args.existing).read_text(encoding="utf-8"),
                                                  Path(args.section).read_text(encoding="utf-8")), encoding="utf-8")


if __name__ == "__main__":
    main()
