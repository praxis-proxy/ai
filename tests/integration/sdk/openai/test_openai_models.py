#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "openai>=2.0,<4",
#     "pytest>=8.0",
# ]
# ///
"""OpenAI SDK compatibility tests for the principal-aware model catalog.

Boots Praxis with the model-catalog example. Both upstream clusters point at
small local servers that only answer GET /v1/models, so the catalog is built
from discovered model lists and any forwarded catalog request would fail.
Asserts that the official OpenAI Python SDK can list and retrieve models, that
the listing depends on the Basic Auth subject, that configured entries overlay
discovered models, and that a hidden model raises the same NotFoundError as an
unknown one.

Usage:
    cargo build -p praxis-ai-proxy --features basic-auth-filter,model-catalog-filter
    uv run tests/integration/sdk/openai/test_openai_models.py -v
"""

from __future__ import annotations

import base64
import json
import os
import re
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from openai import NotFoundError, OpenAI

REPO_ROOT = Path(__file__).resolve().parents[4]
CONFIG_PATH = REPO_ROOT / "examples/configs/model-catalog.yaml"

OPENAI_MODELS = [
    {"id": "gpt-4o", "object": "model", "created": 1715367049, "owned_by": "system"},
    {"id": "gpt-4o-mini", "object": "model", "created": 1721172741, "owned_by": "system"},
]
VLLM_MODELS = [
    {"id": "meta-llama/Llama-3.1-70B-Instruct", "object": "model", "created": 1735000000, "owned_by": "vllm"},
]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _find_binary() -> str:
    configured = os.environ.get("PRAXIS_AI_BIN")
    if configured:
        if os.path.isfile(configured):
            return configured
        raise FileNotFoundError(f"PRAXIS_AI_BIN={configured!r} not found")
    for candidate in ("target/debug/praxis-ai", "target/release/praxis-ai"):
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        "praxis-ai binary not found — run "
        "`cargo build -p praxis-ai-proxy --features basic-auth-filter,model-catalog-filter` first"
    )


def _wait_for_port(port: int, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.1)
    raise TimeoutError(f"port {port} did not accept connections within {timeout}s")


def _example_password() -> str:
    match = re.search(r'password:\s*"([^"]+)"', CONFIG_PATH.read_text())
    assert match, "model-catalog.yaml must declare a Basic Auth password"
    return match.group(1)


def _start_model_list_server(models: list[dict]) -> ThreadingHTTPServer:
    """Serve ``models`` from GET /v1/models and 404 everything else."""
    body = json.dumps({"object": "list", "data": models}).encode()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - http.server API
            if self.path != "/v1/models":
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def _patched_config(listener_port: int, openai_port: int, vllm_port: int) -> str:
    text = CONFIG_PATH.read_text()
    text = text.replace("127.0.0.1:8080", f"127.0.0.1:{listener_port}")
    text = text.replace("127.0.0.1:3000", f"127.0.0.1:{openai_port}")
    text = text.replace("127.0.0.1:3001", f"127.0.0.1:{vllm_port}")
    return text


@pytest.fixture(scope="session")
def praxis_proxy():
    openai = _start_model_list_server(OPENAI_MODELS)
    vllm = _start_model_list_server(VLLM_MODELS)
    listener_port = _free_port()
    fd, config_path = tempfile.mkstemp(suffix=".yaml")
    with os.fdopen(fd, "w") as handle:
        handle.write(_patched_config(listener_port, openai.server_port, vllm.server_port))

    proc = subprocess.Popen(
        [_find_binary(), "-c", config_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        _wait_for_port(listener_port)
        yield listener_port
    finally:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        os.unlink(config_path)
        openai.shutdown()
        vllm.shutdown()


def _client(port: int, username: str) -> OpenAI:
    token = base64.b64encode(f"{username}:{_example_password()}".encode()).decode()
    return OpenAI(
        api_key="not-needed",
        base_url=f"http://127.0.0.1:{port}/v1",
        default_headers={"Authorization": f"Basic {token}"},
        max_retries=0,
        timeout=10.0,
    )


@pytest.fixture(scope="session")
def viewer(praxis_proxy: int) -> OpenAI:
    return _client(praxis_proxy, "viewer")


@pytest.fixture(scope="session")
def ml_engineer(praxis_proxy: int) -> OpenAI:
    return _client(praxis_proxy, "ml-engineer")


class TestOpenAIModels:
    def test_list_is_filtered_per_principal(self, viewer: OpenAI, ml_engineer: OpenAI) -> None:
        assert [m.id for m in viewer.models.list()] == ["gpt-4o", "gpt4o", "gpt-4o-mini"]
        assert [m.id for m in ml_engineer.models.list()] == ["gpt-4o", "gpt4o", "gpt-4o-mini", "llama-70b"]

    def test_list_parses_openai_model_objects(self, viewer: OpenAI) -> None:
        model = next(iter(viewer.models.list()))
        assert model.object == "model"
        assert model.created == 1715367049, "created comes from the upstream"
        assert model.owned_by == "system", "owned_by comes from the upstream"
        extra = model.model_extra or {}
        assert extra["praxis"]["capabilities"] == ["chat_completions", "responses"]
        assert extra["praxis"]["aliases"] == ["gpt4o"]

    def test_unconfigured_upstream_model_is_listed(self, viewer: OpenAI) -> None:
        model = viewer.models.retrieve("gpt-4o-mini")
        extra = model.model_extra or {}
        assert extra["praxis"] == {"provider": "openai"}, "no capability is claimed for unconfigured models"

    def test_retrieve_visible_model(self, ml_engineer: OpenAI) -> None:
        model = ml_engineer.models.retrieve("llama-70b")
        assert model.id == "llama-70b"
        assert model.owned_by == "vllm"

    def test_upstream_id_is_not_listed(self, ml_engineer: OpenAI) -> None:
        with pytest.raises(NotFoundError):
            ml_engineer.models.retrieve("meta-llama/Llama-3.1-70B-Instruct")

    def test_retrieve_alias(self, viewer: OpenAI) -> None:
        model = viewer.models.retrieve("gpt4o")
        assert (model.model_extra or {})["praxis"]["alias_of"] == "gpt-4o"

    def test_hidden_model_looks_unknown(self, viewer: OpenAI) -> None:
        with pytest.raises(NotFoundError) as hidden:
            viewer.models.retrieve("llama-70b")
        with pytest.raises(NotFoundError) as unknown:
            viewer.models.retrieve("no-such-model")
        assert hidden.value.code == unknown.value.code == "model_not_found"
        assert hidden.value.type == unknown.value.type == "invalid_request_error"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, *sys.argv[1:]]))
