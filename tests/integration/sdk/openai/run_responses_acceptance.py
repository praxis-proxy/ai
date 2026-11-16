# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx>=0.27", "pytest>=8.0", "pyyaml>=6.0"]
# ///
"""Run EVERY pinned OpenResponses template against a deployed Praxis endpoint.

Unlike the translation coverage gate, this runner excludes no templates and
fails on skips. Run serially to avoid starving a small inference backend.
Credentials are read from OPENRESPONSES_API_KEY, never command-line arguments.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import signal
import tempfile
from urllib.parse import urlsplit

import yaml

from test_responses_conformance import (
    REPO_ROOT, SUITE_REPO, SUITE_SHA, _ensure_suite,
    _find_binary, _free_port, _wait_for_proxy,
)


def configure_gateway(profile: str, upstream_url: str, model: str, port: int, directory: Path) -> dict:
    """Patch only transport/model settings; keep the versioned filter pipeline."""
    upstream = urlsplit(upstream_url)
    if (upstream.scheme not in {"http", "https"} or not upstream.hostname
            or upstream.username or upstream.password or upstream.query or upstream.fragment
            or upstream.path not in {"", "/", "/v1", "/v1/"}):
        raise ValueError("upstream URL must be HTTP(S), with optional /v1 and no credentials")
    source = REPO_ROOT / f"tests/conformance/openresponses/{profile}.yaml"
    config = yaml.safe_load(source.read_text())
    config["listeners"][0]["address"] = f"127.0.0.1:{port}"

    def patch(value):
        if isinstance(value, dict):
            if "endpoints" in value:
                value["endpoints"] = [upstream.netloc]
                # A URL without a port still needs an explicit socket port.
                if upstream.port is None:
                    host = f"[{upstream.hostname}]" if ":" in upstream.hostname else upstream.hostname
                    value["endpoints"] = [f"{host}:{443 if upstream.scheme == 'https' else 80}"]
                value["http"] = {"authority": upstream.netloc}
                if upstream.scheme == "https":
                    value["tls"] = {"sni": upstream.hostname}
            if "inference_url" in value:
                value["inference_url"] = f"{upstream.scheme}://{upstream.netloc}/v1/chat/completions"
                value["default_model"] = model
            if "database_url" in value:
                value["database_url"] = f"sqlite://{directory}/responses.db?mode=rwc"
            for child in value.values():
                patch(child)
        elif isinstance(value, list):
            for child in value:
                patch(child)

    patch(config)
    return config


@contextmanager
def gateway(profile: str, upstream_url: str, model: str):
    with tempfile.TemporaryDirectory(prefix="praxis-openresponses-") as name:
        directory = Path(name)
        port = _free_port()
        config = configure_gateway(profile, upstream_url, model, port, directory)
        path = directory / "praxis.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        log_path = directory / "praxis.log"
        binary = _find_binary()
        with Path(binary).open("rb") as executable:
            binary_digest = hashlib.file_digest(executable, "sha256").hexdigest()
        with log_path.open("w") as log:
            process = subprocess.Popen([binary, "-c", str(path)], stdout=log, stderr=subprocess.STDOUT)
            try:
                _wait_for_proxy(port, process, log_path)
                yield f"http://127.0.0.1:{port}/v1", {
                    "profile": profile, "binary_sha256": binary_digest,
                    "config": redact(config, urlsplit(upstream_url).netloc),
                }
            finally:
                process.send_signal(signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def enumerate_templates(suite_dir: Path) -> list[dict]:
    """Read the exported templates, including which ones use local fixtures."""
    result = subprocess.run(
        ["bun", "--eval", (
            'import {testTemplates} from "./src/lib/compliance-tests.ts"; '
            'console.log(JSON.stringify(testTemplates.map(t => ({'
            'id:t.id,kind:t.getMockResponse?"schema_fixture":"endpoint",'
            'transport:t.transport??"http"}))))'
        )],
        cwd=suite_dir, capture_output=True, text=True, check=True,
    )
    templates = json.loads(result.stdout)
    ids = [entry["id"] for entry in templates]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("upstream template inventory is empty or has duplicate ids")
    return templates


def run_template(suite_dir: Path, base_url: str, model: str, template: dict) -> dict:
    template_id = template["id"]
    try:
        completed = subprocess.run(
            ["bun", "run", "bin/compliance-test.ts", "--base-url", base_url,
             "--model", model, "--filter", template_id, "--json"],
            cwd=suite_dir, capture_output=True, text=True, timeout=360,
        )
        payload = json.loads(completed.stdout)
        results = payload["results"]
        if len(results) != 1 or results[0]["id"] != template_id:
            raise ValueError(f"expected exactly one result for {template_id}")
        result = dict(results[0])
        if result.get("status") not in {"passed", "failed", "skipped"}:
            raise ValueError("upstream returned an unknown test status")
        if completed.returncode != 0 and result["status"] != "failed":
            raise ValueError(f"upstream CLI exited {completed.returncode}")
    except (OSError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired) as error:
        result = {"id": template_id, "status": "failed", "errors": [str(error)]}
    return redact(
        {**result, "kind": template["kind"], "transport": template["transport"]},
        os.environ.get("OPENRESPONSES_API_KEY", ""),
    )


def redact(value, secret: str):
    """An upstream error or response may echo the supplied credential."""
    if isinstance(value, str):
        return value.replace(secret, "<REDACTED>") if secret else value
    if isinstance(value, list):
        return [redact(item, secret) for item in value]
    if isinstance(value, dict):
        return {key: redact(item, secret) for key, item in value.items()}
    return value


def summarize(results: list[dict]) -> dict:
    counts = {status: sum(r["status"] == status for r in results)
              for status in ("passed", "failed", "skipped")}
    return {**counts, "total": len(results),
            "all_passed": bool(results) and counts["passed"] == len(results)}


def run_acceptance(args, base_url: str, gateway_info: dict) -> int:
    secret = os.environ["OPENRESPONSES_API_KEY"]
    suite_dir = _ensure_suite()
    templates = enumerate_templates(suite_dir)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True))
    report = {
        "suite": {"repo": SUITE_REPO, "commit": SUITE_SHA},
        "bun": subprocess.check_output(["bun", "--version"], text=True).strip(),
        "praxis": {"commit": head, "dirty": dirty, **gateway_info},
        "started_at": datetime.now(timezone.utc).isoformat(),
        "base_url": base_url, "model": args.model,
        "backend": args.backend, "evidence": args.evidence,
        "inventory": templates, "results": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for template in templates:
        result = run_template(suite_dir, base_url, args.model, template)
        report["results"].append(redact(result, secret))
        report["summary"] = summarize(report["results"])
        report["complete"] = len(report["results"]) == len(templates)
        report["summary"]["all_passed"] &= report["complete"]
        args.output.write_text(json.dumps(redact(report, secret), indent=2) + "\n")
        print(f"{template['id']}: {result['status']}", flush=True)
    print(json.dumps(report["summary"]))
    return 0 if report["summary"]["all_passed"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--base-url", help="Already running Praxis URL, including /v1")
    target.add_argument("--upstream-url", help="Launch local Praxis against this backend URL")
    parser.add_argument("--profile", choices=("managed", "passthrough"), default="managed")
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", required=True, help="Backend version/model provenance")
    parser.add_argument("--evidence", choices=("live", "controlled"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    url = urlsplit(args.base_url or args.upstream_url)
    if (url.scheme not in {"http", "https"} or not url.hostname
            or url.username or url.password or url.query or url.fragment):
        parser.error("base URL must be HTTP(S) without credentials, query, or fragment")
    secret = os.environ.get("OPENRESPONSES_API_KEY", "")
    if not secret:
        parser.error("set OPENRESPONSES_API_KEY (use a dummy value for an anonymous local listener)")
    if args.upstream_url:
        with gateway(args.profile, args.upstream_url, args.model) as (base_url, info):
            return run_acceptance(args, base_url, info)
    return run_acceptance(args, args.base_url, {})


if __name__ == "__main__":
    raise SystemExit(main())
