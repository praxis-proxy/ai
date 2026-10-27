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
ports nothing listens on, so every assertion also shows the catalog answers
without contacting a provider. Asserts that the official OpenAI Python SDK can
list and retrieve models, that the listing depends on the Basic Auth subject,
and that a hidden model raises the same NotFoundError as an unknown one.

Usage:
    cargo build -p praxis-ai-proxy --features basic-auth-filter,model-catalog-filter
    uv run tests/integration/sdk/openai/test_openai_models.py -v
"""

from __future__ import annotations

import base64
import os
import re
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest
from openai import NotFoundError, OpenAI

REPO_ROOT = Path(__file__).resolve().parents[4]
CONFIG_PATH = REPO_ROOT / "examples/configs/model-catalog.yaml"


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


def _patched_config(listener_port: int) -> str:
    text = CONFIG_PATH.read_text()
    text = text.replace("127.0.0.1:8080", f"127.0.0.1:{listener_port}")
    # Unused ports: any upstream contact fails instead of silently succeeding.
    text = text.replace("127.0.0.1:3000", f"127.0.0.1:{_free_port()}")
    text = text.replace("127.0.0.1:3001", f"127.0.0.1:{_free_port()}")
    return text


@pytest.fixture(scope="session")
def praxis_proxy():
    listener_port = _free_port()
    fd, config_path = tempfile.mkstemp(suffix=".yaml")
    with os.fdopen(fd, "w") as handle:
        handle.write(_patched_config(listener_port))

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
        assert [m.id for m in viewer.models.list()] == ["gpt-4o", "gpt4o"]
        assert [m.id for m in ml_engineer.models.list()] == ["gpt-4o", "gpt4o", "claude-sonnet-4-5"]

    def test_list_parses_openai_model_objects(self, viewer: OpenAI) -> None:
        model = next(iter(viewer.models.list()))
        assert model.object == "model"
        assert model.created == 1735689600
        assert model.owned_by == "openai", "owned_by defaults to the provider"
        extra = model.model_extra or {}
        assert extra["praxis"]["capabilities"] == ["chat_completions", "responses"]
        assert extra["praxis"]["aliases"] == ["gpt4o"]

    def test_retrieve_visible_model(self, ml_engineer: OpenAI) -> None:
        model = ml_engineer.models.retrieve("claude-sonnet-4-5")
        assert model.id == "claude-sonnet-4-5"
        assert model.owned_by == "anthropic"

    def test_retrieve_alias(self, viewer: OpenAI) -> None:
        model = viewer.models.retrieve("gpt4o")
        assert (model.model_extra or {})["praxis"]["alias_of"] == "gpt-4o"

    def test_hidden_model_looks_unknown(self, viewer: OpenAI) -> None:
        with pytest.raises(NotFoundError) as hidden:
            viewer.models.retrieve("claude-sonnet-4-5")
        with pytest.raises(NotFoundError) as unknown:
            viewer.models.retrieve("no-such-model")
        assert hidden.value.code == unknown.value.code == "model_not_found"
        assert hidden.value.type == unknown.value.type == "invalid_request_error"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, *sys.argv[1:]]))
