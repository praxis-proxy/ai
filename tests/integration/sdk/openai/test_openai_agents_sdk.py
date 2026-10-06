#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "openai-agents==0.23.1",
#     "openai>=3.0.0,<4",
#     "pytest>=8.0",
# ]
# ///
"""OpenAI Agents SDK compatibility test for the Praxis Responses endpoint.

Runs a complete, client-owned function-tool loop through the OpenAI Agents SDK
runner (`Runner`) against a local Praxis listener. The Agents SDK uses the
Responses API by default; this proves the SDK-managed model -> tool -> model
loop works end to end through Praxis's `POST /v1/responses`, closing the
framework-level compatibility evidence gap tracked in
https://github.com/praxis-proxy/ai/issues/1592.

This is NOT OpenAI Agents API support. The test must never call `/v1/agents/*`
or Chat Completions; a scripted Responses backend sits behind Praxis and the
router only publishes `/v1/responses`, so any other path fails loudly.

Determinism: the backend is scripted, so the model "decides" to call the tool
regardless of inference probability. Two model rounds are hard-scripted:
  1. first Response emits a `function_call` for `get_weather`;
  2. the Agents SDK executes the local Python tool exactly once;
  3. the continuation request carries the matching `function_call_output`;
  4. the backend returns a final message with a unique marker.

The test uses no credentials, no external network, no retries, and no xfail.
Agents SDK tracing is disabled so the runner cannot export traces to OpenAI.

Usage:
    cargo build -p praxis-ai-proxy
    uv run tests/integration/sdk/openai/test_openai_agents_sdk.py -s
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

# Disable Agents SDK tracing before importing the package so the runner never
# attempts to export traces or reach OpenAI. Belt and suspenders: the env var,
# the in-code switch below, and a dummy key so no real credential is required.
os.environ.setdefault("OPENAI_AGENTS_DISABLE_TRACING", "1")
os.environ.setdefault("OPENAI_API_KEY", "sk-praxis-agents-sdk-not-needed")

from agents import (  # noqa: E402  (import after env setup is intentional)
    Agent,
    OpenAIResponsesModel,
    Runner,
    function_tool,
    set_default_openai_api,
    set_tracing_disabled,
)
from openai import AsyncOpenAI  # noqa: E402

set_tracing_disabled(True)
# Keep the Responses provider path explicit even though the Agent below binds an
# OpenAIResponsesModel directly. The Agents SDK must not fall back to Chat
# Completions for any model it constructs on our behalf.
set_default_openai_api("responses")

REPO_ROOT = Path(__file__).resolve().parents[4]
CONFIG_PATH = REPO_ROOT / "examples/configs/openai/responses/responses-proxy.yaml"

PINNED_AGENTS_VERSION = "0.23.1"

MODEL = "praxis-test-model"
EXPECTED_CITY = "Boston"
# Unique, deterministic markers prove the payloads crossed Praxis unchanged.
TOOL_OUTPUT_MARKER = "WEATHER-TOOL-RESULT::b4e1f6a2"
FINAL_OUTPUT_MARKER = "AGENTS-SDK-LOOP-OK::7f3c9a2e5d"
CALL_ID = "call_praxis_weather_1"

# Record every invocation of the local tool so the test can assert it ran
# exactly once with the expected arguments.
TOOL_CALLS: list[dict[str, Any]] = []


@function_tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    TOOL_CALLS.append({"city": city})
    return f"{TOOL_OUTPUT_MARKER}::{city}"


# -----------------------------------------------------------------------------
# Scripted native Responses backend
# -----------------------------------------------------------------------------


def _usage() -> dict[str, Any]:
    return {
        "input_tokens": 11,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": 7,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": 18,
    }


def _response_envelope(resp_id: str, output: list[dict[str, Any]]) -> dict[str, Any]:
    """A complete native /v1/responses body the OpenAI SDK parses into Response.

    Mirrors a recorded native Responses payload (see
    tests/integration/fixtures/inference/recordings/vllm/responses/) so lenient
    SDK construction finds every field it reads.
    """
    return {
        "id": resp_id,
        "object": "response",
        "created_at": 0,
        "status": "completed",
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "max_output_tokens": None,
        "metadata": None,
        "model": MODEL,
        "output": output,
        "parallel_tool_calls": True,
        "previous_response_id": None,
        "prompt": None,
        "reasoning": None,
        "service_tier": "auto",
        "temperature": 1.0,
        "text": None,
        "tool_choice": "auto",
        "tools": [],
        "top_p": 1.0,
        "truncation": "disabled",
        "usage": _usage(),
        "user": None,
    }


def _function_call_response() -> dict[str, Any]:
    return _response_envelope(
        "resp_round_1",
        [
            {
                "type": "function_call",
                "id": "fc_praxis_weather_1",
                "call_id": CALL_ID,
                "name": "get_weather",
                "arguments": json.dumps({"city": EXPECTED_CITY}),
                "status": "completed",
            }
        ],
    )


def _final_message_response() -> dict[str, Any]:
    return _response_envelope(
        "resp_round_2",
        [
            {
                "type": "message",
                "id": "msg_praxis_final_1",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": FINAL_OUTPUT_MARKER,
                        "annotations": [],
                    }
                ],
            }
        ],
    )


class _Backend:
    """Scripts native Responses bodies and records every forwarded request."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.requests: list[tuple[str, dict[str, Any]]] = []
        self.scripts: deque[dict[str, Any]] = deque()

    def reset(self) -> None:
        with self.lock:
            self.requests.clear()
            self.scripts.clear()

    def script(self, body: dict[str, Any]) -> None:
        with self.lock:
            self.scripts.append(body)

    def record(self, path: str, body: dict[str, Any]) -> None:
        with self.lock:
            self.requests.append((path, body))

    def take(self) -> dict[str, Any] | None:
        with self.lock:
            if self.scripts:
                return self.scripts.popleft()
        return None


_STATE = _Backend()


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args: Any) -> None:  # silence access logs
        pass

    def do_GET(self) -> None:
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.send_header("Connection", "close")
        self.end_headers()

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b""
        try:
            parsed = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            parsed = {"__unparsed__": raw.decode("utf-8", "replace")}
        _STATE.record(self.path, parsed)

        scripted = _STATE.take()
        if scripted is None:
            payload = json.dumps(
                {"error": {"message": "scripted backend exhausted"}}
            ).encode()
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(payload)
            return

        payload = json.dumps(scripted).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(payload)


# -----------------------------------------------------------------------------
# Praxis lifecycle (self-contained; mirrors the sibling SDK suites)
# -----------------------------------------------------------------------------


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
        "praxis-ai binary not found — run `cargo build -p praxis-ai-proxy` first"
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


def _patched_config(listener_port: int, backend_port: int) -> str:
    text = CONFIG_PATH.read_text()
    text = text.replace("127.0.0.1:8080", f"127.0.0.1:{listener_port}")
    text = text.replace("127.0.0.1:3001", f"127.0.0.1:{backend_port}")
    return text


@pytest.fixture(scope="session")
def praxis_proxy() -> Any:
    backend_port = _free_port()
    listener_port = _free_port()
    server = ThreadingHTTPServer(("127.0.0.1", backend_port), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    fd, config_path = tempfile.mkstemp(suffix=".yaml")
    with os.fdopen(fd, "w") as handle:
        handle.write(_patched_config(listener_port, backend_port))

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
        server.shutdown()
        os.unlink(config_path)


@pytest.fixture
def agents_model(praxis_proxy: int) -> OpenAIResponsesModel:
    client = AsyncOpenAI(
        api_key="sk-praxis-agents-sdk-not-needed",
        base_url=f"http://127.0.0.1:{praxis_proxy}/v1",
        max_retries=0,
        timeout=10.0,
    )
    # Explicit Responses provider: the runner must speak /v1/responses only.
    return OpenAIResponsesModel(model=MODEL, openai_client=client)


@pytest.fixture(autouse=True)
def _reset_backend() -> None:
    _STATE.reset()
    TOOL_CALLS.clear()


# -----------------------------------------------------------------------------
# Deterministic acceptance test (required CI gate)
# -----------------------------------------------------------------------------


def _find_item(items: list[dict[str, Any]], item_type: str) -> dict[str, Any]:
    for item in items:
        if isinstance(item, dict) and item.get("type") == item_type:
            return item
    raise AssertionError(f"no {item_type!r} item in {items!r}")


class TestAgentsSdkResponsesLoop:
    def test_function_tool_loop_over_responses(
        self, agents_model: OpenAIResponsesModel
    ) -> None:
        _STATE.script(_function_call_response())
        _STATE.script(_final_message_response())

        agent = Agent(
            name="weather-agent",
            instructions="Use the get_weather tool to answer weather questions.",
            tools=[get_weather],
            model=agents_model,
        )

        result = Runner.run_sync(agent, "What is the weather in Boston?")

        # The SDK returned the scripted final message as its final output.
        assert result.final_output == FINAL_OUTPUT_MARKER

        # Exactly two model rounds occurred, both to /v1/responses — never
        # /v1/agents/* and never Chat Completions.
        paths = [path for path, _ in _STATE.requests]
        assert len(paths) == 2, f"expected exactly two model rounds, got {paths}"
        assert all(p == "/v1/responses" for p in paths), paths
        assert not any("/v1/agents" in p for p in paths), paths
        assert not any("/v1/chat/completions" in p for p in paths), paths

        # The local tool executed exactly once with the expected arguments.
        assert TOOL_CALLS == [{"city": EXPECTED_CITY}], TOOL_CALLS

        # Round 1: the declared function schema crossed Praxis correctly.
        _, round1 = _STATE.requests[0]
        tools = round1.get("tools")
        assert isinstance(tools, list) and len(tools) == 1, round1
        tool = tools[0]
        assert tool["type"] == "function", tool
        assert tool["name"] == "get_weather", tool
        params = tool["parameters"]
        assert "city" in params["properties"], params
        assert params["properties"]["city"]["type"] == "string", params
        assert "city" in params["required"], params

        # Round 2: the tool output crossed Praxis as the matching
        # function_call_output in the continuation request.
        _, round2 = _STATE.requests[1]
        assert isinstance(round2.get("input"), list), round2
        fco = _find_item(round2["input"], "function_call_output")
        assert fco["call_id"] == CALL_ID, fco
        expected_output = f"{TOOL_OUTPUT_MARKER}::{EXPECTED_CITY}"
        assert fco["output"] == expected_output, fco

    def test_pinned_versions_recorded(self) -> None:
        agents_version = importlib.metadata.version("openai-agents")
        openai_version = importlib.metadata.version("openai")
        print(
            f"\n[agents-sdk] openai-agents=={agents_version} openai=={openai_version}",
            file=sys.stderr,
        )
        assert agents_version == PINNED_AGENTS_VERSION, (
            f"openai-agents resolved to {agents_version}, expected the pinned "
            f"{PINNED_AGENTS_VERSION}; update the PEP 723 header and this pin together"
        )
        assert openai_version.startswith("3."), (
            f"openai resolved to {openai_version}; the Agents SDK requires the 3.x lane"
        )


# Record the pinned versions at import time so `uv run ... -s` always prints
# them in CI logs even before pytest reporting begins.
print(
    "[agents-sdk] pinned openai-agents=="
    f"{importlib.metadata.version('openai-agents')} "
    f"openai=={importlib.metadata.version('openai')}",
    file=sys.stderr,
)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"] + sys.argv[1:]))
