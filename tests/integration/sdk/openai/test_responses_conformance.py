# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx>=0.27", "pytest>=8.0", "pyyaml>=6.0"]
# ///
"""OpenResponses conformance runner for the openai_responses_to_chat_completions filter.

Boots a Praxis listener that translates POST /v1/responses -> POST
/v1/chat/completions, fronts a vLLM CPU backend with a translation-witness
shim, and runs the pinned OpenResponses Bun suite against the supported
templates. Never exercises native Responses passthrough.
"""

from __future__ import annotations

import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
CONFIG_PATH = REPO_ROOT / "examples/configs/openai/responses/responses-to-chat-completions.yaml"
MANIFEST_PATH = REPO_ROOT / "tests/conformance/openresponses/manifest.yaml"
OVERLAY_PATH = REPO_ROOT / "tests/conformance/openresponses/package.json"

SUITE_REPO = "https://github.com/openresponses/openresponses"
SUITE_SHA = "92c12d96d7b61d6d15e2214daa5e9c6000ab6e1c"

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen3-0.6B")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _find_binary() -> str:
    override = os.environ.get("PRAXIS_AI_BIN")
    if override:
        if os.path.isfile(override):
            return override
        raise FileNotFoundError(f"PRAXIS_AI_BIN={override!r} not found")
    for candidate in ("target/debug/praxis-ai", "target/release/praxis-ai"):
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError("praxis-ai binary not found — run `cargo build -p praxis-ai-proxy` first")


def _load_manifest() -> dict:
    return yaml.safe_load(MANIFEST_PATH.read_text())


def _manifest_ids(manifest: dict) -> tuple[set[str], set[str], set[str]]:
    supported = {entry["id"] for entry in manifest["supported"]}
    unsupported = {entry["id"] for entry in manifest["unsupported"]}
    inapplicable = {entry["id"] for entry in manifest["inapplicable"]}
    return supported, unsupported, inapplicable


def _ensure_suite() -> Path:
    suite_dir = Path(
        os.environ.get("OPENRESPONSES_SUITE_DIR")
        or (Path(os.environ.get("RUNNER_TEMP", "/tmp")) / f"openresponses-{SUITE_SHA}")
    )
    if not (suite_dir / "bin" / "compliance-test.ts").exists():
        suite_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "init", "-q"], cwd=suite_dir, check=True)
        subprocess.run(["git", "remote", "add", "origin", SUITE_REPO], cwd=suite_dir, check=True)
        subprocess.run(["git", "fetch", "-q", "--depth", "1", "origin", SUITE_SHA], cwd=suite_dir, check=True)
        subprocess.run(["git", "checkout", "-q", "FETCH_HEAD"], cwd=suite_dir, check=True)
    # Overlay a zod-only package.json so `bun install` skips the site tree.
    shutil.copyfile(OVERLAY_PATH, suite_dir / "package.json")
    lock = suite_dir / "bun.lock"
    if lock.exists():
        lock.unlink()
    subprocess.run(["bun", "install"], cwd=suite_dir, check=True)
    return suite_dir


def _enumerate_template_ids(suite_dir: Path) -> set[str]:
    import re

    text = (suite_dir / "src/lib/compliance-tests.ts").read_text()
    return set(re.findall(r'^    id: "(.+?)",?$', text, re.MULTILINE))


def _run_compliance(suite_dir: Path, listener_port: int, ids: set[str]) -> subprocess.CompletedProcess:
    cmd = [
        "bun", "run", "bin/compliance-test.ts",
        "--base-url", f"http://127.0.0.1:{listener_port}/v1",
        "--api-key", "test",
        "--model", VLLM_MODEL,
        "--filter", ",".join(sorted(ids)),
        "--json",
    ]
    return subprocess.run(cmd, cwd=suite_dir, capture_output=True, text=True)


class _WitnessHandler(BaseHTTPRequestHandler):
    seen_paths: list[tuple[str, str]] = []

    def log_message(self, *_args):  # silence access logging
        pass

    def _forward(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b""
        type(self).seen_paths.append((self.command, self.path))
        headers = {k: v for k, v in self.headers.items() if k.lower() not in ("host", "content-length")}
        url = f"{VLLM_BASE_URL}{self.path}"
        with httpx.Client(timeout=300.0) as client:
            with client.stream(self.command, url, headers=headers, content=body) as upstream:
                self.send_response(upstream.status_code)
                for key, value in upstream.headers.items():
                    if key.lower() in ("transfer-encoding", "content-length", "connection"):
                        continue
                    self.send_header(key, value)
                self.end_headers()
                for chunk in upstream.iter_raw():
                    if chunk:
                        self.wfile.write(chunk)
                        self.wfile.flush()

    def do_POST(self):
        self._forward()

    def do_GET(self):
        self._forward()


def _start_witness() -> tuple[ThreadingHTTPServer, int, list[tuple[str, str]]]:
    _WitnessHandler.seen_paths = []
    port = _free_port()
    server = ThreadingHTTPServer(("127.0.0.1", port), _WitnessHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, port, _WitnessHandler.seen_paths


def _patched_config(listener_port: int, backend_port: int, db_path: Path) -> str:
    text = CONFIG_PATH.read_text()
    text = text.replace("127.0.0.1:8080", f"127.0.0.1:{listener_port}")
    text = text.replace("127.0.0.1:3001", f"127.0.0.1:{backend_port}")
    # The shared example persists via a store filter; isolate its SQLite file in
    # the test's tmp dir so the run leaves nothing behind and never contends on a
    # repo-root responses.db. The conformance suite is stateless, so the store is
    # a no-op for correctness — this only redirects where it writes.
    text = text.replace("sqlite://responses.db?mode=rwc", f"sqlite://{db_path}?mode=rwc")
    return text


def _wait_for_proxy(port: int, proc: subprocess.Popen, log_path: Path, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        exit_code = proc.poll()
        if exit_code is not None:
            raise RuntimeError(f"praxis-ai exited with code {exit_code} before binding {port}:\n{log_path.read_text()}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.2)
    raise TimeoutError(f"praxis-ai did not start within {timeout}s on port {port}:\n{log_path.read_text()}")


@pytest.fixture()
def praxis_proxy(tmp_path):
    witness, witness_port, seen = _start_witness()
    listener_port = _free_port()
    config_file = tmp_path / "conformance.yaml"
    config_file.write_text(_patched_config(listener_port, witness_port, tmp_path / "responses.db"))
    log_path = tmp_path / "praxis.log"
    with open(log_path, "w") as log_file:
        proc = subprocess.Popen([_find_binary(), "-c", str(config_file)], stdout=log_file, stderr=subprocess.STDOUT)
        try:
            _wait_for_proxy(listener_port, proc, log_path)
            yield listener_port, seen
        finally:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
            witness.shutdown()


def test_suite_template_ids_are_fully_triaged():
    suite_dir = _ensure_suite()
    supported, unsupported, inapplicable = _manifest_ids(_load_manifest())
    enumerated = _enumerate_template_ids(suite_dir)
    triaged = supported | unsupported | inapplicable
    assert enumerated == triaged, (
        f"untriaged templates: {sorted(enumerated - triaged)}; "
        f"stale manifest ids: {sorted(triaged - enumerated)}"
    )


def test_openresponses_conformance_supported_all_pass(praxis_proxy):
    # In CI the composite action's readiness step guarantees vLLM is up before
    # this runs, so this skip only fires for local runs without a backend.
    try:
        httpx.get(f"{VLLM_BASE_URL}/v1/models", timeout=2.0)
    except Exception:
        pytest.skip(f"vLLM not reachable at {VLLM_BASE_URL}")
    listener_port, seen = praxis_proxy
    suite_dir = _ensure_suite()
    supported, _, _ = _manifest_ids(_load_manifest())
    result = _run_compliance(suite_dir, listener_port, supported)
    assert result.returncode == 0, f"suite failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    paths = {(method, path) for method, path in seen}
    assert ("POST", "/v1/chat/completions") in paths, f"backend never saw chat completions; saw {paths}"
    assert not any(path.startswith("/v1/responses") for _method, path in seen), (
        f"criterion (f) violated: backend saw native Responses traffic: {seen}"
    )


def test_openresponses_conformance_unsupported_stays_failing(praxis_proxy):
    # Promotion nudge: `unsupported` templates are in-scope translation gaps we
    # do not pass yet. If a translator fix makes one pass, this test turns red
    # and tells you to promote it — so the coverage bump can never go unnoticed.
    # Runs each id on its own so the nudge names the exact template to promote.
    try:
        httpx.get(f"{VLLM_BASE_URL}/v1/models", timeout=2.0)
    except Exception:
        pytest.skip(f"vLLM not reachable at {VLLM_BASE_URL}")
    listener_port, _seen = praxis_proxy
    suite_dir = _ensure_suite()
    _, unsupported, _ = _manifest_ids(_load_manifest())
    now_passing = [
        template_id
        for template_id in sorted(unsupported)
        if _run_compliance(suite_dir, listener_port, {template_id}).returncode == 0
    ]
    assert not now_passing, (
        "these `unsupported` templates now PASS — the translator gained a capability. "
        "Promote them to `supported` in tests/conformance/openresponses/manifest.yaml "
        "and regenerate the report (`cargo xtask openresponses-coverage --fix`); "
        f"coverage will bump: {now_passing}"
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
