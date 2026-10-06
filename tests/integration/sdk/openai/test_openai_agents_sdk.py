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

Determinism (default): the backend is scripted, so the model "decides" to call
the tool regardless of inference probability. Two model rounds are hard-scripted:
  1. first Response emits a `function_call` for `get_weather`;
  2. the Agents SDK executes the local Python tool exactly once;
  3. the continuation request carries the matching `function_call_output`;
  4. the backend returns a final message with a unique marker.

The default mode uses no credentials, no external network, no retries, and no
xfail. Agents SDK tracing is disabled so the runner cannot export traces to
OpenAI.

Live mode (GPU suite): set `VLLM_TEST_BACKEND=live` to run the identical loop
against a real vLLM serving `/v1/responses` (`VLLM_MODEL` names the served
model, `PRAXIS_TEST_VLLM_BASE_URL` points at it). The local recorder stays on
the wire and forwards to vLLM, so the "only `/v1/responses` crossed Praxis" and
`function_call_output` assertions still hold; the tool call is forced via
`tool_choice="required"` so a real model's probability cannot flake the loop.

Usage:
    # Deterministic (scripted backend), the required CI gate:
    cargo build -p praxis-ai-proxy
    uv run tests/integration/sdk/openai/test_openai_agents_sdk.py -s

    # Live vLLM (GPU suite):
    VLLM_TEST_BACKEND=live VLLM_MODEL=Qwen/Qwen3-8B \
      PRAXIS_TEST_VLLM_BASE_URL=http://127.0.0.1:8000 \
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
import urllib.error
import urllib.request
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
    ModelSettings,
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

# Live mode (GPU suite): instead of the scripted backend, the recorder forwards
# every request to a real vLLM serving /v1/responses, so an actual model decides
# to call the tool. Selected by VLLM_TEST_BACKEND=live; VLLM_MODEL names the
# served model and PRAXIS_TEST_VLLM_BASE_URL points at the running vLLM. The
# recorder stays in the path so the "only /v1/responses crossed Praxis" and
# function_call_output assertions still hold against the real backend.
LIVE = os.environ.get("VLLM_TEST_BACKEND") == "live"
VLLM_BASE_URL = os.environ.get("PRAXIS_TEST_VLLM_BASE_URL", "http://127.0.0.1:8000")
# In live mode the served model name must match vLLM's --served-model-name.
ACTIVE_MODEL = os.environ.get("VLLM_MODEL", MODEL) if LIVE else MODEL
# A real model load + generation is far slower than the scripted backend.
REQUEST_TIMEOUT = 300.0 if LIVE else 10.0

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

    def _relay(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        if body:
            self.wfile.write(body)

    def _forward(self, raw: bytes) -> None:
        """Live mode: forward the request to the real vLLM and relay its reply.

        The recorder stays on the wire (Praxis -> recorder -> vLLM) so the
        request-inspection assertions hold against a real backend.
        """
        url = VLLM_BASE_URL.rstrip("/") + self.path
        req = urllib.request.Request(url, data=raw, method="POST")
        req.add_header(
            "Content-Type", self.headers.get("Content-Type", "application/json")
        )
        auth = self.headers.get("Authorization")
        if auth:
            req.add_header("Authorization", auth)
        try:
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                body = resp.read()
                self._relay(
                    resp.status,
                    body,
                    resp.headers.get("Content-Type", "application/json"),
                )
        except urllib.error.HTTPError as exc:
            body = exc.read()
            self._relay(
                exc.code, body, exc.headers.get("Content-Type", "application/json")
            )
        except (urllib.error.URLError, OSError) as exc:
            # A transport failure talking to live vLLM (refused, reset, read
            # timeout) would otherwise escape do_POST, drop the socket, and
            # surface as an opaque proxy 5xx. Relay a legible 504 so the cause
            # lands in the GPU step's tee'd log instead.
            body = json.dumps(
                {"error": {"message": f"recorder->vLLM forward failed: {exc}"}}
            ).encode()
            self._relay(504, body, "application/json")

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

        if LIVE:
            self._forward(raw)
            return

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


# A loaded GPU runner can be slow to cold-start the full debug binary, so the
# default startup budget is generous and overridable from CI.
PROXY_START_TIMEOUT = float(os.environ.get("PRAXIS_TEST_PROXY_START_TIMEOUT", "30"))


def _proxy_log_tail(log_path: str | None, limit: int = 4000) -> str:
    if not log_path:
        return "(no proxy output captured)"
    try:
        text = Path(log_path).read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return "(no proxy output captured)"
    return text[-limit:] if text else "(proxy produced no output)"


def _wait_for_port(
    port: int, proc: subprocess.Popen, log_path: str | None, timeout: float
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        exit_code = proc.poll()
        if exit_code is not None:
            raise RuntimeError(
                f"praxis-ai exited with code {exit_code} before binding port "
                f"{port}; output:\n{_proxy_log_tail(log_path)}"
            )
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.1)
    raise TimeoutError(
        f"port {port} did not accept connections within {timeout}s; "
        f"praxis-ai output:\n{_proxy_log_tail(log_path)}"
    )


def _patched_config(listener_port: int, backend_port: int) -> str:
    text = CONFIG_PATH.read_text()
    text = text.replace("127.0.0.1:8080", f"127.0.0.1:{listener_port}")
    text = text.replace("127.0.0.1:3001", f"127.0.0.1:{backend_port}")
    # CI GPU runners execute as root, and Praxis refuses to start as root unless
    # the config opts in. Enable it in this throwaway test copy only; the shipped
    # example stays strict. On a non-root host the flag is a harmless no-op.
    if "allow_root:" not in text:
        if "insecure_options:" in text:
            text = text.replace(
                "insecure_options:", "insecure_options:\n  allow_root: true", 1
            )
        else:
            text += "\ninsecure_options:\n  allow_root: true\n"
    return text


def _stop_proc(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def _unlink(*paths: str | None) -> None:
    for path in paths:
        if path:
            try:
                os.unlink(path)
            except OSError:
                pass


@pytest.fixture(scope="session")
def praxis_proxy() -> Any:
    backend_port = _free_port()
    server = ThreadingHTTPServer(("127.0.0.1", backend_port), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    binary = _find_binary()
    proc: subprocess.Popen | None = None
    config_path: str | None = None
    log_path: str | None = None
    errors: list[str] = []
    try:
        # Retry a few times: _free_port() races with other processes on a busy
        # runner, and a lost race makes praxis-ai exit with "address in use".
        for attempt in range(3):
            listener_port = _free_port()
            fd, config_path = tempfile.mkstemp(suffix=".yaml")
            with os.fdopen(fd, "w") as handle:
                handle.write(_patched_config(listener_port, backend_port))
            log_fd, log_path = tempfile.mkstemp(suffix=".log")
            with os.fdopen(log_fd, "wb") as log_file:
                proc = subprocess.Popen(
                    [binary, "-c", config_path],
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                )
            try:
                _wait_for_port(listener_port, proc, log_path, PROXY_START_TIMEOUT)
                break
            except (TimeoutError, RuntimeError) as exc:
                errors.append(f"attempt {attempt + 1}: {exc}")
                _stop_proc(proc)
                proc = None
                _unlink(config_path, log_path)
                config_path = log_path = None
        else:
            raise RuntimeError("praxis-ai did not start:\n" + "\n".join(errors))
        yield listener_port
    finally:
        _stop_proc(proc)
        server.shutdown()
        _unlink(config_path, log_path)


@pytest.fixture
def agents_model(praxis_proxy: int) -> OpenAIResponsesModel:
    client = AsyncOpenAI(
        api_key="sk-praxis-agents-sdk-not-needed",
        base_url=f"http://127.0.0.1:{praxis_proxy}/v1",
        max_retries=0,
        timeout=REQUEST_TIMEOUT,
    )
    # Explicit Responses provider: the runner must speak /v1/responses only.
    return OpenAIResponsesModel(model=ACTIVE_MODEL, openai_client=client)


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
        model_kwargs: dict[str, Any] = {}
        if LIVE:
            # Force the real model to call the tool so the loop is deterministic;
            # reset_tool_choice (Agent default True) flips tool_choice back to
            # "auto" after the call, so the continuation round is free-form and
            # cannot loop forever.
            model_kwargs["model_settings"] = ModelSettings(
                tool_choice="required", max_tokens=512
            )
        else:
            # The scripted backend drives both rounds; no live model is involved.
            _STATE.script(_function_call_response())
            _STATE.script(_final_message_response())

        agent = Agent(
            name="weather-agent",
            instructions="Use the get_weather tool to answer weather questions.",
            tools=[get_weather],
            model=agents_model,
            **model_kwargs,
        )

        prompt = "What is the weather in Boston?"
        if LIVE:
            # Qwen3 serves a reasoning parser; skip thinking for a fast, stable run.
            prompt += " Use the get_weather tool. /no_think"

        result = Runner.run_sync(agent, prompt)

        # The loop produced a final textual answer.
        assert isinstance(result.final_output, str) and result.final_output, result

        # Every model round crossed Praxis as POST /v1/responses — never
        # /v1/agents/* and never Chat Completions.
        paths = [path for path, _ in _STATE.requests]
        assert paths, "no requests reached the backend"
        assert all(p == "/v1/responses" for p in paths), paths
        assert not any("/v1/agents" in p for p in paths), paths
        assert not any("/v1/chat/completions" in p for p in paths), paths

        # The local tool executed with the expected city.
        assert TOOL_CALLS, "get_weather was never called"
        assert all("boston" in c["city"].lower() for c in TOOL_CALLS), TOOL_CALLS

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

        # Continuation round: the tool output crossed Praxis as the matching
        # function_call_output. Its payload is the exact value the local tool
        # returned, and its call_id links back to the model's function_call.
        continuation = next(
            (
                body
                for _, body in _STATE.requests
                if isinstance(body.get("input"), list)
                and any(
                    isinstance(item, dict)
                    and item.get("type") == "function_call_output"
                    for item in body["input"]
                )
            ),
            None,
        )
        assert continuation is not None, _STATE.requests
        fco = _find_item(continuation["input"], "function_call_output")
        assert fco["call_id"], fco
        called_city = TOOL_CALLS[0]["city"]
        assert fco["output"] == f"{TOOL_OUTPUT_MARKER}::{called_city}", fco

        if LIVE:
            # A real model drives the rounds: at least the initial request plus
            # one continuation carrying the tool output.
            assert len(paths) >= 2, paths
        else:
            # Scripted backend: exactly two rounds, the pinned call id, and the
            # unique final marker returned verbatim.
            assert len(paths) == 2, f"expected exactly two model rounds, got {paths}"
            assert TOOL_CALLS == [{"city": EXPECTED_CITY}], TOOL_CALLS
            assert fco["call_id"] == CALL_ID, fco
            assert result.final_output == FINAL_OUTPUT_MARKER

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
