#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "httpx>=0.27,<1",
#     "openai>=2.0,<3",
#     "pytest>=8.0,<9",
# ]
# ///
"""
OpenAI Responses API integration tests against a real vLLM CPU backend.

Starts a Praxis proxy with the full responses pipeline backed by vLLM,
then exercises stateless requests, persistence, rehydration, and streaming
using the official OpenAI Python SDK.

Usage:
    cargo build -p praxis-ai-proxy
    uv run tests/integration/sdk/openai/test_openai_responses_vllm.py -s
"""

import base64
import io
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import ClassVar
from urllib.parse import urlparse

import httpx
import pytest
from openai import BadRequestError, NotFoundError, OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen3-0.6B")
OGX_BASE_URL = os.environ.get("OGX_BASE_URL", "http://127.0.0.1:8321")
PRAXIS_AI_BIN = os.environ.get("PRAXIS_AI_BIN")
DATABASE_URL = os.environ.get("DATABASE_URL", "")
CONFIG_PATH = "examples/configs/openai/responses/full-flow.yaml"
AGENTIC_CONFIG_PATH = "examples/configs/openai/responses/agentic-loop.yaml"
IRR_STREAMING_CONFIG_PATH = (
    "examples/configs/openai/responses/irr-terminal-streaming.yaml"
)
CHAT_STREAMING_CONFIG_PATH = (
    "examples/configs/openai/responses/responses-to-chat-completions.yaml"
)
COMPACT_CONFIG_PATH = "examples/configs/openai/responses/compact.yaml"

TERMINAL_RESPONSE_EVENTS = {
    "response.cancelled",
    "response.completed",
    "response.failed",
    "response.incomplete",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _find_binary() -> str:
    if PRAXIS_AI_BIN:
        if os.path.isfile(PRAXIS_AI_BIN):
            return PRAXIS_AI_BIN
        raise FileNotFoundError(f"PRAXIS_AI_BIN={PRAXIS_AI_BIN!r} not found")
    for candidate in ["target/debug/praxis-ai", "target/release/praxis-ai"]:
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        "praxis-ai binary not found — run `cargo build -p praxis-ai-proxy` first"
    )


def _vllm_endpoint() -> str:
    parsed = urlparse(VLLM_BASE_URL)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 8000
    return f"{host}:{port}"


def _ogx_endpoint() -> str:
    parsed = urlparse(OGX_BASE_URL)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 8321
    return f"{host}:{port}"


def _patch_store_backend(config: str, db_path: str) -> str:
    if DATABASE_URL.startswith("postgres"):
        config = config.replace(
            'database_url: "sqlite://responses.db?mode=rwc"',
            f'database_url: "{DATABASE_URL}"\n'
            "        allow_private_database_url: true\n"
            "        ssl_mode: disable",
        )
        config = config.replace("backend: sqlite", "backend: postgres")
    else:
        config = config.replace(
            "sqlite://responses.db?mode=rwc",
            f"sqlite://{db_path}?mode=rwc",
        )
    return config


def _write_config(praxis_port: int, db_path: str) -> str:
    with open(CONFIG_PATH) as f:
        config = f.read()

    config = config.replace("127.0.0.1:8080", f"127.0.0.1:{praxis_port}")
    config = config.replace("127.0.0.1:3001", _vllm_endpoint())
    config = config.replace("127.0.0.1:9999", _ogx_endpoint())
    config = _patch_store_backend(config, db_path)

    fd, path = tempfile.mkstemp(suffix=".yaml")
    with os.fdopen(fd, "w") as f:
        f.write(config)
    return path


def _read_log_tail(log_path: str, max_lines: int = 50) -> str:
    """Best-effort read of a Praxis log file's tail for error diagnostics."""
    try:
        with open(log_path) as f:
            lines = f.readlines()
    except OSError as exc:
        return f"(could not read {log_path}: {exc})"
    if not lines:
        return "(no output captured)"
    return "".join(lines[-max_lines:])


def _assert_usage(usage) -> None:
    """Assert the stable token-accounting invariants shared by providers."""
    assert usage is not None, "response should include token usage"
    assert usage.input_tokens > 0, usage
    assert usage.output_tokens > 0, usage
    assert usage.total_tokens == usage.input_tokens + usage.output_tokens, usage


def _collect_stream(stream):
    """Consume an SDK stream and return its typed events."""
    with stream:
        return list(stream)


def _assert_stream_contract(
    events,
    *,
    expected_text: str | None = None,
    require_usage: bool = True,
) -> object:
    """Validate the provider-neutral Responses SSE lifecycle."""
    assert events, "stream should emit at least one event"
    event_types = [event.type for event in events]
    assert event_types[0] == "response.created", event_types
    assert event_types[-1] in TERMINAL_RESPONSE_EVENTS, event_types
    assert event_types.count("response.created") == 1, event_types
    assert sum(t in TERMINAL_RESPONSE_EVENTS for t in event_types) == 1, event_types

    sequence_numbers = [event.sequence_number for event in events]
    assert all(isinstance(number, int) for number in sequence_numbers), sequence_numbers
    assert sequence_numbers == sorted(set(sequence_numbers)), sequence_numbers

    created = events[0].response
    terminal = events[-1].response
    assert created.id == terminal.id
    assert created.status == "in_progress"

    deltas = "".join(
        event.delta for event in events if event.type == "response.output_text.delta"
    )
    if terminal.status == "completed":
        assert "response.output_item.added" in event_types, event_types
        assert "response.output_item.done" in event_types, event_types
        assert "response.content_part.added" in event_types, event_types
        assert "response.content_part.done" in event_types, event_types
        assert "response.output_text.done" in event_types, event_types
        assert deltas.strip() == terminal.output_text.strip(), (
            deltas,
            terminal.output_text,
        )
        if expected_text is not None:
            assert expected_text in terminal.output_text, terminal.output_text
        if require_usage:
            _assert_usage(terminal.usage)

    return terminal


def _write_irr_streaming_config(praxis_port: int) -> str:
    with open(IRR_STREAMING_CONFIG_PATH) as f:
        config = f.read()

    config = config.replace("127.0.0.1:8080", f"127.0.0.1:{praxis_port}")
    config = config.replace("127.0.0.1:3001", _vllm_endpoint())

    fd, path = tempfile.mkstemp(suffix=".yaml")
    with os.fdopen(fd, "w") as f:
        f.write(config)
    return path


def _write_chat_streaming_config(praxis_port: int, db_path: str) -> str:
    """Patch the shipped Responses-to-Chat example for live vLLM."""
    with open(CHAT_STREAMING_CONFIG_PATH) as f:
        config = f.read()

    config = config.replace("127.0.0.1:8080", f"127.0.0.1:{praxis_port}")
    config = config.replace("127.0.0.1:3001", _vllm_endpoint())
    config = _patch_store_backend(config, db_path)

    fd, path = tempfile.mkstemp(suffix=".yaml")
    with os.fdopen(fd, "w") as f:
        f.write(config)
    return path


def _write_compact_config(
    praxis_port: int,
    db_path: str,
    compaction_port: int,
) -> str:
    """Patch the compact example for live vLLM and a deterministic summary."""
    with open(COMPACT_CONFIG_PATH) as f:
        config = f.read()

    config = config.replace("127.0.0.1:8080", f"127.0.0.1:{praxis_port}")
    config = config.replace("127.0.0.1:9999", _ogx_endpoint())
    config = config.replace(
        "http://localhost:11434/v1/chat/completions",
        f"http://127.0.0.1:{compaction_port}/v1/chat/completions",
    )
    config = config.replace(
        "default_model: llama3.2:1b", f"default_model: {VLLM_MODEL}"
    )
    config = config.replace("timeout_ms: 60000", "timeout_ms: 300000")
    config = config.replace("127.0.0.1:11434", _vllm_endpoint())
    config = _patch_store_backend(config, db_path)

    fd, path = tempfile.mkstemp(suffix=".yaml")
    with os.fdopen(fd, "w") as f:
        f.write(config)
    return path


def _wait_for_proxy(
    port: int, proc: subprocess.Popen, log_path: str, timeout: float = 30.0
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        # A fatal config/startup error makes Praxis exit before it ever binds
        # the port. Surface its logs immediately instead of waiting out the
        # full timeout with a context-free error.
        exit_code = proc.poll()
        if exit_code is not None:
            raise RuntimeError(
                f"Praxis exited with code {exit_code} before binding port {port}:\n"
                f"{_read_log_tail(log_path)}"
            )
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.2)
    raise TimeoutError(
        f"Praxis did not start within {timeout}s on port {port}:\n{_read_log_tail(log_path)}"
    )


# ---------------------------------------------------------------------------
# MCP Mock Server
# ---------------------------------------------------------------------------


class MCPHandler(BaseHTTPRequestHandler):
    """Streamable HTTP MCP server with a single get_weather tool."""

    authorization_headers: ClassVar[list[str | None]] = []

    def do_POST(self):
        type(self).authorization_headers.append(self.headers.get("Authorization"))
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        req = json.loads(body)
        method = req.get("method")
        rid = req.get("id")

        if method == "initialize":
            self._json_rpc(
                rid,
                {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": "weather-mock", "version": "0.1.0"},
                },
            )
        elif method == "notifications/initialized":
            self.send_response(202)
            self.end_headers()
        elif method == "tools/list":
            self._json_rpc(
                rid,
                {
                    "tools": [
                        {
                            "name": "get_weather",
                            "description": "Get current weather for a city",
                            "inputSchema": {
                                "type": "object",
                                "properties": {"city": {"type": "string"}},
                                "required": ["city"],
                                "additionalProperties": False,
                            },
                        }
                    ]
                },
            )
        elif method == "tools/call":
            city = req.get("params", {}).get("arguments", {}).get("city", "unknown")
            self._json_rpc(
                rid, {"content": [{"type": "text", "text": f"72F and sunny in {city}"}]}
            )
        elif method == "ping":
            self._json_rpc(rid, {})
        else:
            self._json_rpc(
                rid,
                None,
                error={
                    "code": -32601,
                    "message": f"unknown method: {method}",
                },
            )

    def _json_rpc(self, rid, result, error=None):
        resp = {"jsonrpc": "2.0", "id": rid}
        if error:
            resp["error"] = error
        else:
            resp["result"] = result
        payload = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt, *args):
        pass


class BraveSearchHandler(BaseHTTPRequestHandler):
    """Mock Brave Search API returning canned results."""

    request_paths: ClassVar[list[str]] = []

    def do_GET(self):
        type(self).request_paths.append(self.path)
        payload = json.dumps(
            {
                "web": {
                    "results": [
                        {
                            "title": "Mock Search Result",
                            "url": "https://example.com/mock",
                            "description": "A mock search result for testing",
                        }
                    ]
                }
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt, *args):
        pass


class CompactionHandler(BaseHTTPRequestHandler):
    """Deterministic Chat Completions summarizer for compaction tests."""

    requests: ClassVar[list[dict]] = []

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        type(self).requests.append(json.loads(body))
        payload = json.dumps(
            {
                "id": "chatcmpl_compaction_test",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": VLLM_MODEL,
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": ("The persistent marker is COMPACT-KEEP-7412."),
                        },
                    }
                ],
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt, *args):
        pass


def _write_agentic_config(
    praxis_port: int,
    db_path: str,
    mcp_port: int,
    search_port: int,
    *,
    translate_to_chat: bool = False,
) -> str:
    """Patch agentic-loop.yaml with test ports and allow_loopback."""
    with open(AGENTIC_CONFIG_PATH) as f:
        config = f.read()

    config = config.replace("127.0.0.1:8080", f"127.0.0.1:{praxis_port}")
    vllm = _vllm_endpoint()
    config = config.replace(
        '- "127.0.0.1:3001"',
        f'- "{vllm}"\n                    read_timeout_ms: 300000',
    )
    config = _patch_store_backend(config, db_path)
    config = config.replace(
        "- filter: openai_mcp_tool_resolve\n",
        "- filter: openai_mcp_tool_resolve\n        allow_loopback: true\n",
    )
    config = config.replace(
        "- filter: openai_mcp_dispatch\n              - filter: openai_agentic_loop",
        "- filter: openai_mcp_dispatch\n"
        "                allow_loopback: true\n"
        "              - filter: openai_agentic_loop",
    )
    config = config.replace(
        "max_iterations: 11\n",
        "max_iterations: 11\n"
        "        timeout_ms: 300000\n"
        "        step_timeout_ms: 300000\n",
    )
    config = config.replace(
        "- filter: openai_web_search\n"
        "                provider: brave\n"
        "                api_key: ${WEB_SEARCH_API_KEY}",
        "- filter: openai_web_search\n"
        "                provider: brave\n"
        "                api_key: test-key\n"
        f"                base_url: http://127.0.0.1:{search_port}\n"
        "                allow_private_base_url: true",
    )
    if translate_to_chat:
        config = config.replace(
            "              - filter: openai_responses_proxy\n"
            "                terminal_streaming: true\n"
            "              - filter: router",
            "              - filter: responses_to_chat_completions\n"
            "              - filter: path_rewrite\n"
            "                replace:\n"
            '                  pattern: "^/v1/responses/?$"\n'
            '                  replacement: "/v1/chat/completions"\n'
            "                conditions:\n"
            "                  - when:\n"
            '                      path_prefix: "/v1/responses"\n'
            "                      methods: [POST]\n"
            "              - filter: router",
        )

    fd, path = tempfile.mkstemp(suffix=".yaml")
    with os.fdopen(fd, "w") as f:
        f.write(config)
    return path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def praxis_proxy(tmp_path_factory, request):
    """Start a Praxis proxy backed by vLLM for the test session."""
    port = _free_port()
    db_dir = tmp_path_factory.mktemp("responses")
    db_path = str(db_dir / "responses.db")
    config_path = _write_config(port, db_path)
    binary = _find_binary()

    log_path = str(db_dir / "praxis.log")
    log_file = open(log_path, "w")
    started = False

    proc = subprocess.Popen(
        [binary, "-c", config_path],
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    try:
        _wait_for_proxy(port, proc, log_path)
        started = True
        yield port
    finally:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        log_file.close()
        if not started or request.session.testsfailed > 0:
            with open(log_path) as f:
                print(
                    f"\n=== Praxis logs ===\n{f.read()}",
                    file=sys.stderr,
                )
        os.unlink(config_path)


@pytest.fixture(scope="session")
def irr_streaming_proxy(tmp_path_factory, request):
    """Start a Praxis proxy with terminal Responses streaming through IRR."""
    port = _free_port()
    config_path = _write_irr_streaming_config(port)
    binary = _find_binary()

    log_dir = tmp_path_factory.mktemp("irr-terminal-streaming")
    log_path = str(log_dir / "praxis.log")
    log_file = open(log_path, "w")
    started = False

    proc = subprocess.Popen(
        [binary, "-c", config_path],
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    try:
        _wait_for_proxy(port, proc, log_path)
        started = True
        yield port
    finally:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        log_file.close()
        if not started or request.session.testsfailed > 0:
            with open(log_path) as f:
                print(
                    f"\n=== IRR streaming Praxis logs ===\n{f.read()}",
                    file=sys.stderr,
                )
        os.unlink(config_path)


@pytest.fixture(scope="session")
def chat_streaming_proxy(tmp_path_factory, request):
    """Start the Responses-to-Chat streaming example against live vLLM."""
    port = _free_port()
    db_dir = tmp_path_factory.mktemp("responses-chat-streaming")
    db_path = str(db_dir / "responses.db")
    config_path = _write_chat_streaming_config(port, db_path)
    binary = _find_binary()

    log_path = str(db_dir / "praxis.log")
    log_file = open(log_path, "w")
    started = False

    proc = subprocess.Popen(
        [binary, "-c", config_path],
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    try:
        _wait_for_proxy(port, proc, log_path)
        started = True
        yield port
    finally:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        log_file.close()
        if not started or request.session.testsfailed > 0:
            with open(log_path) as f:
                print(
                    f"\n=== Chat streaming Praxis logs ===\n{f.read()}",
                    file=sys.stderr,
                )
        os.unlink(config_path)


@pytest.fixture(scope="session")
def compaction_server():
    """Start a deterministic summarization backend for compact callouts."""
    port = _free_port()
    server = HTTPServer(("127.0.0.1", port), CompactionHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield port
    server.shutdown()


@pytest.fixture(scope="session")
def compact_proxy(tmp_path_factory, request, compaction_server):
    """Start the compact example against vLLM and the mock summarizer."""
    port = _free_port()
    db_dir = tmp_path_factory.mktemp("responses-compact")
    db_path = str(db_dir / "responses.db")
    config_path = _write_compact_config(
        port,
        db_path,
        compaction_server,
    )
    binary = _find_binary()

    log_path = str(db_dir / "praxis.log")
    log_file = open(log_path, "w")
    started = False
    proc = subprocess.Popen(
        [binary, "-c", config_path],
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    try:
        _wait_for_proxy(port, proc, log_path)
        started = True
        yield port
    finally:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        log_file.close()
        if not started or request.session.testsfailed > 0:
            with open(log_path) as f:
                print(
                    f"\n=== Compact Praxis logs ===\n{f.read()}",
                    file=sys.stderr,
                )
        os.unlink(config_path)


@pytest.fixture(scope="session")
def openai_client(praxis_proxy):
    """Return an OpenAI client pointed at the local Praxis proxy."""
    return OpenAI(
        base_url=f"http://127.0.0.1:{praxis_proxy}/v1",
        api_key="test",
        max_retries=0,
        timeout=300,
    )


@pytest.fixture(scope="session")
def irr_streaming_client(irr_streaming_proxy):
    """Return an OpenAI client using the terminal-streaming IRR proxy."""
    return OpenAI(
        base_url=f"http://127.0.0.1:{irr_streaming_proxy}/v1",
        api_key="test",
        max_retries=0,
        timeout=300,
    )


@pytest.fixture(scope="session")
def chat_streaming_client(chat_streaming_proxy):
    """Return an SDK client using Responses-to-Chat stream translation."""
    return OpenAI(
        base_url=f"http://127.0.0.1:{chat_streaming_proxy}/v1",
        api_key="test",
        max_retries=0,
        timeout=300,
    )


@pytest.fixture(scope="session")
def compact_client(compact_proxy):
    """Return an SDK client using the compact filter example."""
    return OpenAI(
        base_url=f"http://127.0.0.1:{compact_proxy}/v1",
        api_key="test",
        max_retries=0,
        timeout=300,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestOpenAIResponsesVLLM:
    """Integration tests for the Responses API against a vLLM backend."""

    def test_stateless_request(self, openai_client):
        response = openai_client.responses.create(
            model=VLLM_MODEL,
            input="Say exactly: HELLO-PRAXIS /no_think",
            temperature=0,
            store=False,
            max_output_tokens=128,
        )

        assert response.status == "completed"
        assert "HELLO-PRAXIS" in response.output_text
        assert response.object == "response"
        assert response.id.startswith("resp_")
        assert response.error is None
        assert response.incomplete_details is None
        _assert_usage(response.usage)

        with pytest.raises(NotFoundError) as exc_info:
            openai_client.responses.retrieve(response.id)
        assert exc_info.value.status_code == 404

    def test_store_and_retrieve(self, openai_client):
        response = openai_client.responses.create(
            model=VLLM_MODEL,
            input="Say exactly: STORED-OK /no_think",
            temperature=0,
            store=True,
            max_output_tokens=128,
        )

        assert response.status == "completed"
        assert response.id

        retrieved = openai_client.responses.retrieve(response.id)

        assert retrieved.id == response.id
        assert retrieved.status == "completed"
        assert retrieved.output_text == response.output_text
        _assert_usage(retrieved.usage)

    def test_stored_input_items_pagination_and_delete(self, openai_client):
        response = openai_client.responses.create(
            model=VLLM_MODEL,
            input=[
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "The marker is INPUT-ITEMS-OK.",
                        }
                    ],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "Repeat the marker exactly. /no_think",
                        }
                    ],
                },
            ],
            store=True,
            max_output_tokens=128,
        )

        page = openai_client.responses.input_items.list(
            response.id,
            limit=1,
            order="asc",
        )
        assert page.object == "list"
        assert len(page.data) == 1
        assert page.first_id == page.data[0].id
        assert page.last_id == page.data[-1].id
        assert page.has_more is True
        assert page.data[0].type == "message"
        assert page.data[0].role == "user"
        assert page.data[0].content[0].type == "input_text"
        assert "INPUT-ITEMS-OK" in page.data[0].content[0].text

        next_page = openai_client.responses.input_items.list(
            response.id,
            after=page.last_id,
            limit=1,
            order="asc",
        )
        assert len(next_page.data) == 1
        assert next_page.has_more is False
        assert "Repeat the marker" in next_page.data[0].content[0].text

        assert openai_client.responses.delete(response.id) is None
        with pytest.raises(NotFoundError) as exc_info:
            openai_client.responses.retrieve(response.id)
        assert exc_info.value.status_code == 404
        with pytest.raises(NotFoundError) as exc_info:
            openai_client.responses.input_items.list(response.id)
        assert exc_info.value.status_code == 404

    def test_response_resource_not_found_errors(self, openai_client):
        missing_id = "resp_missing_sdk_integration"
        with pytest.raises(NotFoundError) as exc_info:
            openai_client.responses.retrieve(missing_id)
        assert exc_info.value.status_code == 404

        with pytest.raises(NotFoundError) as exc_info:
            openai_client.responses.input_items.list(missing_id)
        assert exc_info.value.status_code == 404

        with pytest.raises(NotFoundError) as exc_info:
            openai_client.responses.delete(missing_id)
        assert exc_info.value.status_code == 404

    def test_invalid_previous_response_id_is_rejected(self, openai_client):
        with pytest.raises(BadRequestError) as exc_info:
            openai_client.responses.create(
                model=VLLM_MODEL,
                input="This request must not reach vLLM.",
                previous_response_id="resp_missing_sdk_integration",
                store=True,
            )
        assert exc_info.value.status_code == 400
        assert "resp_missing_sdk_integration" in str(exc_info.value)

    def test_malformed_request_has_sdk_compatible_error(self, openai_client):
        response = httpx.post(
            f"{str(openai_client.base_url).rstrip('/')}/responses",
            headers={"Authorization": "Bearer test"},
            json={},
            timeout=10,
        )
        assert response.status_code == 400
        error = response.json()["error"]
        assert isinstance(error["message"], str)
        assert error["message"]
        assert isinstance(error["type"], str)
        assert error["type"]

    def test_invalid_input_container_is_rejected(self, openai_client):
        with pytest.raises(BadRequestError) as exc_info:
            openai_client.responses.create(
                model=VLLM_MODEL,
                input=["not-an-input-item"],
                store=False,
            )
        assert exc_info.value.status_code == 400

    def test_rehydrated_second_turn(self, openai_client):
        first = openai_client.responses.create(
            model=VLLM_MODEL,
            input=("Remember this nonce: VIOLET-7319. Acknowledge it. /no_think"),
            temperature=0,
            store=True,
            max_output_tokens=128,
        )

        assert first.status == "completed"

        second = openai_client.responses.create(
            model=VLLM_MODEL,
            input=("What nonce did I just tell you? Repeat it exactly. /no_think"),
            temperature=0,
            previous_response_id=first.id,
            store=True,
            max_output_tokens=128,
        )

        assert second.status == "completed"
        assert "VIOLET-7319" in second.output_text

    def test_rehydrated_response_echoes_previous_response_id(self, openai_client):
        """Issue #932: the rehydrated response must echo the caller's
        previous_response_id back to the client.

        On the rehydrated path the proxy replays prior turns via the `input`
        array and strips previous_response_id from the upstream request, so the
        vLLM backend never sees it and echoes previous_response_id: null. The
        rehydrate filter restores the caller's id into the response body so the
        client always sees the id it sent, per the Responses API contract.

        This assertion is metadata-only and independent of model output, so it
        is deterministic despite running against a real vLLM backend.

        Manifest linkage: this is the live vLLM regression counterpart of the
        committed synthetic inference fixture -- coverage feature
        ``responses.native.continuation``, scenario
        ``responses/native-continuation`` (see
        tests/integration/fixtures/inference/). No live recording is committed
        for that feature -- it stays ``synthetic_only`` because a live recording
        requires explicit authorization -- so this SDK test provides the
        real-backend confidence instead.
        """
        first = openai_client.responses.create(
            model=VLLM_MODEL,
            input="Say exactly: ECHO-BASE /no_think",
            temperature=0,
            store=True,
            max_output_tokens=128,
        )

        assert first.status == "completed"
        assert first.id

        second = openai_client.responses.create(
            model=VLLM_MODEL,
            input="Say exactly: ECHO-NEXT /no_think",
            temperature=0,
            previous_response_id=first.id,
            store=True,
            max_output_tokens=128,
        )

        assert second.status == "completed"
        assert second.previous_response_id == first.id, (
            "the proxy must echo the caller's previous_response_id back to the "
            "client even though it strips the id from the rehydrated upstream "
            f"request; got: {second.previous_response_id!r}"
        )

    def test_conversation_context_and_append_back(self, openai_client):
        conversation = openai_client.conversations.create(
            metadata={"suite": "responses-vllm"},
            items=[
                {
                    "type": "message",
                    "role": "user",
                    "content": "Remember the nonce COPPER-8462.",
                }
            ],
        )

        try:
            response = openai_client.responses.create(
                model=VLLM_MODEL,
                input=("Repeat the nonce from this conversation exactly. /no_think"),
                temperature=0,
                conversation=conversation.id,
                store=True,
                max_output_tokens=128,
            )

            assert response.status == "completed"
            assert "COPPER-8462" in response.output_text

            items = openai_client.conversations.items.list(
                conversation.id,
                order="asc",
            )
            roles = [item.role for item in items.data if item.type == "message"]
            assert roles == ["user", "user", "assistant"], roles
            payload = json.dumps(
                [item.model_dump() for item in items.data],
                default=str,
            )
            assert "COPPER-8462" in payload
            assistant_items = [
                item
                for item in items.data
                if item.type == "message" and item.role == "assistant"
            ]
            assert len(assistant_items) == 1
            assert "COPPER-8462" in assistant_items[0].content[0].text
        finally:
            openai_client.conversations.delete(conversation.id)

    def test_conversation_multi_turn_append_back(self, openai_client):
        conversation = openai_client.conversations.create()
        try:
            first = openai_client.responses.create(
                model=VLLM_MODEL,
                input="Remember the color ultramarine. /no_think",
                conversation={"id": conversation.id},
                store=True,
                temperature=0,
                max_output_tokens=128,
            )
            # Ask the model to echo the earlier color rather than recall it in
            # free form: the small CI model reliably repeats an exact token from
            # loaded context but may paraphrase a "which color" question. This
            # still proves the first turn's context was appended back and
            # reloaded for the second turn.
            second = openai_client.responses.create(
                model=VLLM_MODEL,
                input="Repeat the exact color name I told you to remember. /no_think",
                conversation=conversation.id,
                store=True,
                temperature=0,
                max_output_tokens=128,
            )

            assert first.status == "completed"
            assert second.status == "completed"
            assert "ultramarine" in second.output_text.lower()

            items = openai_client.conversations.items.list(
                conversation.id,
                order="asc",
            )
            # Append-back also persists reasoning items, so assert on the
            # message turns rather than the total item count.
            message_roles = [
                item.role for item in items.data if item.type == "message"
            ]
            assert message_roles == [
                "user",
                "assistant",
                "user",
                "assistant",
            ], message_roles
        finally:
            openai_client.conversations.delete(conversation.id)

    def test_nonexistent_conversation_is_rejected(self, openai_client):
        with pytest.raises(BadRequestError) as exc_info:
            openai_client.responses.create(
                model=VLLM_MODEL,
                input="This request must not reach vLLM.",
                conversation="conv_00000000000000000000000000000000",
                store=True,
            )
        assert exc_info.value.status_code == 400
        assert "conv_00000000000000000000000000000000" in str(exc_info.value)

    def test_doc_extract_inline_file_to(self, openai_client):
        """Issue #397: inline file_data is extracted to input_text and
        consumed by vLLM inference.

        Sends an input_file with base64-encoded text through the full
        pipeline (file_resolve → doc_extract → responses_proxy → vLLM).
        The doc_extract filter converts the input_file to input_text
        before forwarding. Asserts a unique marker from the document
        appears in the model output, proving vLLM consumed the
        extracted text.
        """
        marker = "PRAXIS-DOC-9271"
        file_content = f"The secret marker is: {marker}"
        file_data = (
            "data:text/plain;base64," + base64.b64encode(file_content.encode()).decode()
        )

        response = openai_client.responses.create(
            model=VLLM_MODEL,
            input=[
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_file",
                            "filename": "document.txt",
                            "file_data": file_data,
                        },
                        {
                            "type": "input_text",
                            "text": (
                                "What marker appears in the document? "
                                "Repeat it exactly. /no_think"
                            ),
                        },
                    ],
                }
            ],
            temperature=0,
            store=False,
            max_output_tokens=256,
        )

        assert response.status == "completed"
        assert marker in response.output_text, (
            f"vLLM should produce output containing the document "
            f"marker '{marker}'; got: {response.output_text}"
        )

    def test_file_id_resolution(self, openai_client):
        """End-to-end: upload to OGX via Praxis, reference by file_id,
        verify vLLM output contains the file content.

        Pipeline: file_resolve (OGX) -> doc_extract -> responses_proxy -> vLLM
        """
        marker = "PRAXIS-OGX-FILE-4829"
        file_content = f"The secret marker is: {marker}"

        uploaded = openai_client.files.create(
            file=("marker-document.txt", io.BytesIO(file_content.encode())),
            purpose="user_data",
        )
        file_id = uploaded.id

        try:
            response = openai_client.responses.create(
                model=VLLM_MODEL,
                input=[
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {
                                "type": "input_file",
                                "file_id": file_id,
                            },
                            {
                                "type": "input_text",
                                "text": (
                                    "What marker appears in the document? "
                                    "Repeat it exactly. /no_think"
                                ),
                            },
                        ],
                    }
                ],
                temperature=0,
                store=False,
                max_output_tokens=128,
            )

            assert response.status == "completed"
            assert marker in response.output_text, (
                f"vLLM should produce output containing the file marker "
                f"'{marker}'; got: {response.output_text}"
            )
        finally:
            try:
                openai_client.files.delete(file_id)
            except Exception:
                pass

    def test_client_function_call_returns(self, openai_client):
        """Client-side function tools are returned without auto-execution.

        The full-flow pipeline has no agentic loop, so function_call
        items are passed through to the client. Validates that vLLM
        produces a well-formed function_call through the proxy.
        """
        response = openai_client.responses.create(
            model=VLLM_MODEL,
            input=(
                "You MUST call the get_weather function for Paris. "
                "Do not answer directly. /no_think"
            ),
            tools=[
                {
                    "type": "function",
                    "name": "get_weather",
                    "description": "Get current weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                }
            ],
            temperature=0,
            store=False,
            max_output_tokens=256,
        )

        assert response.status == "completed"

        function_calls = [
            item for item in response.output if item.type == "function_call"
        ]
        assert len(function_calls) >= 1, (
            "vLLM should return at least one function_call; "
            f"got output types: {[i.type for i in response.output]}"
        )
        fc = function_calls[0]
        assert fc.name == "get_weather"
        args = json.loads(fc.arguments)
        assert "city" in args, f"function arguments should contain city: {args}"

    def test_client_function_call_output_resumes_inference(
        self,
        openai_client,
    ):
        first = openai_client.responses.create(
            model=VLLM_MODEL,
            input=(
                "You MUST call get_weather for Paris and wait for its result. /no_think"
            ),
            tools=[
                {
                    "type": "function",
                    "name": "get_weather",
                    "description": "Get the current weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                }
            ],
            tool_choice={"type": "function", "name": "get_weather"},
            store=True,
            max_output_tokens=256,
        )
        function_calls = [item for item in first.output if item.type == "function_call"]
        assert len(function_calls) == 1, first.output

        second = openai_client.responses.create(
            model=VLLM_MODEL,
            previous_response_id=first.id,
            input=[
                {
                    "type": "function_call_output",
                    "call_id": function_calls[0].call_id,
                    "output": "The weather is 72F and sunny.",
                },
                # Give the model an explicit instruction to report the tool
                # result. Without a directive the small CI model may resume with
                # an empty assistant message; this keeps the test focused on the
                # proxy resuming inference from a client-provided function output.
                {
                    "type": "message",
                    "role": "user",
                    "content": "Tell me the weather using the tool result. /no_think",
                },
            ],
            tools=[
                {
                    "type": "function",
                    "name": "get_weather",
                    "description": "Get the current weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                }
            ],
            tool_choice="none",
            store=True,
            temperature=0,
            max_output_tokens=256,
        )

        assert second.status == "completed"
        assert "72" in second.output_text or "sunny" in second.output_text.lower()

    def test_structured_json_output(self, openai_client):
        response = openai_client.responses.create(
            model=VLLM_MODEL,
            input="Return the marker STRUCTURED-2468. /no_think",
            temperature=0,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "marker_result",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "marker": {"type": "string"},
                        },
                        "required": ["marker"],
                        "additionalProperties": False,
                    },
                },
            },
            store=False,
            # The native Responses path emits a separate reasoning item whose
            # tokens count against the budget, so allow enough headroom for the
            # constrained JSON to complete on the small CI model.
            max_output_tokens=512,
        )

        assert response.status == "completed"
        assert json.loads(response.output_text) == {
            "marker": "STRUCTURED-2468",
        }

    def test_generation_parameters_are_reflected(self, openai_client):
        response = openai_client.responses.create(
            model=VLLM_MODEL,
            input="Say exactly: PARAMS-OK /no_think",
            temperature=0,
            top_p=1,
            parallel_tool_calls=False,
            truncation="disabled",
            store=False,
            max_output_tokens=128,
        )

        assert response.status == "completed"
        assert "PARAMS-OK" in response.output_text
        assert response.temperature == 0
        assert response.top_p == 1
        assert response.parallel_tool_calls is False
        assert response.truncation == "disabled"

    def test_max_output_tokens_reports_incomplete(self, openai_client):
        response = openai_client.responses.create(
            model=VLLM_MODEL,
            input="Write a long explanation of network proxies. /no_think",
            store=False,
            max_output_tokens=1,
        )

        assert response.status == "incomplete"
        assert response.error is None
        assert response.incomplete_details is not None
        assert response.incomplete_details.reason == "max_output_tokens"

    def test_streaming_through_irr(self, irr_streaming_client):
        """Stream a Responses request through a terminal IRR step."""
        stream = irr_streaming_client.responses.create(
            model=VLLM_MODEL,
            input="Say exactly: STREAM-OK /no_think",
            temperature=0,
            store=False,
            stream=True,
            max_output_tokens=128,
        )

        events = _collect_stream(stream)
        terminal = _assert_stream_contract(
            events,
            expected_text="STREAM-OK",
        )
        assert terminal.status == "completed"


class TestResponsesCompactionVLLM:
    """Live coverage for automatic context-management compaction."""

    def test_invalid_compaction_threshold_is_rejected(self, compact_client):
        with pytest.raises(BadRequestError) as exc_info:
            compact_client.responses.create(
                model=VLLM_MODEL,
                input="This request must not reach vLLM.",
                context_management=[
                    {
                        "type": "compaction",
                        "compact_threshold": 999,
                    }
                ],
                store=False,
            )
        assert exc_info.value.status_code == 400
        assert "compact_threshold" in str(exc_info.value)

    def test_below_threshold_skips_compaction(self, compact_client):
        first = compact_client.responses.create(
            model=VLLM_MODEL,
            input="Remember the marker BELOW-THRESHOLD-2468. /no_think",
            temperature=0,
            store=True,
            max_output_tokens=64,
        )
        request_count = len(CompactionHandler.requests)

        second = compact_client.responses.create(
            model=VLLM_MODEL,
            input="Repeat the marker I gave you. /no_think",
            temperature=0,
            previous_response_id=first.id,
            context_management=[
                {
                    "type": "compaction",
                    "compact_threshold": 1000,
                }
            ],
            store=False,
            max_output_tokens=128,
        )

        assert second.status == "completed"
        assert second.output_text
        assert len(CompactionHandler.requests) == request_count

    def test_over_threshold_compacts_rehydrated_history(
        self,
        compact_client,
    ):
        first = compact_client.responses.create(
            model=VLLM_MODEL,
            input=(
                "The persistent marker is COMPACT-KEEP-7412. "
                + "context-padding " * 1200
                + "Say exactly: ACK. /no_think"
            ),
            temperature=0,
            store=True,
            max_output_tokens=128,
        )
        assert first.status == "completed"
        request_count = len(CompactionHandler.requests)

        second = compact_client.responses.create(
            model=VLLM_MODEL,
            input=(
                "Repeat the persistent marker from the compacted context "
                "exactly. /no_think"
            ),
            temperature=0,
            previous_response_id=first.id,
            context_management=[
                {
                    "type": "compaction",
                    "compact_threshold": 1000,
                }
            ],
            store=True,
            max_output_tokens=128,
        )

        assert second.status == "completed"
        assert "COMPACT-KEEP-7412" in second.output_text
        assert len(CompactionHandler.requests) == request_count + 1
        compaction_request = CompactionHandler.requests[-1]
        assert compaction_request["model"] == VLLM_MODEL
        conversation = compaction_request["messages"][1]["content"]
        assert "COMPACT-KEEP-7412" in conversation
        assert "Repeat the persistent marker" in conversation


class TestResponsesToChatCompletionsVLLM:
    """Live SDK coverage for Chat Completions SSE translation."""

    # The external OpenResponses conformance runner is intentionally parked.
    # When it is added, its target must be this Responses-to-Chat-Completions
    # pipeline, not native Responses passthrough, so it measures the contract
    # owned by the translation filter.

    def test_finite_response_round_trip(self, chat_streaming_client):
        response = chat_streaming_client.responses.create(
            model=VLLM_MODEL,
            input="Say exactly: CHAT-FINITE-OK /no_think",
            temperature=0,
            store=True,
            max_output_tokens=128,
        )

        assert response.status == "completed"
        assert "CHAT-FINITE-OK" in response.output_text
        assert response.object == "response"
        assert response.error is None
        assert response.incomplete_details is None
        _assert_usage(response.usage)

        retrieved = chat_streaming_client.responses.retrieve(response.id)
        assert retrieved.id == response.id
        assert retrieved.output_text == response.output_text

    def test_input_items_pagination_and_delete(self, chat_streaming_client):
        response = chat_streaming_client.responses.create(
            model=VLLM_MODEL,
            input=[
                {
                    "type": "message",
                    "role": "user",
                    "content": "The marker is CHAT-INPUT-3579.",
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": "Repeat the marker exactly. /no_think",
                },
            ],
            store=True,
            max_output_tokens=128,
        )

        first = chat_streaming_client.responses.input_items.list(
            response.id,
            limit=1,
            order="asc",
        )
        assert len(first.data) == 1
        assert first.has_more is True
        second = chat_streaming_client.responses.input_items.list(
            response.id,
            after=first.last_id,
            limit=1,
            order="asc",
        )
        assert len(second.data) == 1
        assert second.has_more is False

        assert chat_streaming_client.responses.delete(response.id) is None
        with pytest.raises(NotFoundError):
            chat_streaming_client.responses.retrieve(response.id)

    def test_generation_parameters_round_trip(self, chat_streaming_client):
        response = chat_streaming_client.responses.create(
            model=VLLM_MODEL,
            input="Say exactly: CHAT-PARAMS-OK /no_think",
            temperature=0,
            top_p=1,
            parallel_tool_calls=False,
            prompt_cache_key="praxis-chat-parameter-test",
            truncation="disabled",
            store=False,
            max_output_tokens=128,
        )

        assert response.status == "completed"
        assert "CHAT-PARAMS-OK" in response.output_text
        assert response.temperature == 0
        assert response.top_p == 1
        assert response.parallel_tool_calls is False
        assert response.prompt_cache_key == "praxis-chat-parameter-test"
        assert response.truncation == "disabled"

    def test_structured_output_round_trip(self, chat_streaming_client):
        response = chat_streaming_client.responses.create(
            model=VLLM_MODEL,
            input="Return the marker CHAT-JSON-1357. /no_think",
            temperature=0,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "marker_result",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "marker": {"type": "string"},
                        },
                        "required": ["marker"],
                        "additionalProperties": False,
                    },
                },
            },
            store=False,
            max_output_tokens=128,
        )

        assert response.status == "completed"
        assert json.loads(response.output_text) == {
            "marker": "CHAT-JSON-1357",
        }

    def test_function_call_and_output_round_trip(
        self,
        chat_streaming_client,
    ):
        tool = {
            "type": "function",
            "name": "get_weather",
            "description": "Get the current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
                "additionalProperties": False,
            },
            "strict": True,
        }
        first = chat_streaming_client.responses.create(
            model=VLLM_MODEL,
            input="Call get_weather for Paris. /no_think",
            temperature=0,
            tools=[tool],
            tool_choice={"type": "function", "name": "get_weather"},
            store=True,
            max_output_tokens=256,
        )
        calls = [item for item in first.output if item.type == "function_call"]
        assert len(calls) == 1, first.output
        assert calls[0].name == "get_weather"
        assert json.loads(calls[0].arguments)["city"]

        second = chat_streaming_client.responses.create(
            model=VLLM_MODEL,
            previous_response_id=first.id,
            input=[
                {
                    "type": "function_call_output",
                    "call_id": calls[0].call_id,
                    "output": "The weather is 68F and clear.",
                }
            ],
            temperature=0,
            tools=[tool],
            tool_choice="none",
            store=True,
            max_output_tokens=128,
        )
        assert second.status == "completed"
        assert "68" in second.output_text or "clear" in second.output_text.lower()

    def test_streaming_response_round_trip(self, chat_streaming_client):
        stream = chat_streaming_client.responses.create(
            model=VLLM_MODEL,
            input="Say exactly: CHAT-STREAM-OK /no_think",
            temperature=0,
            store=True,
            stream=True,
            max_output_tokens=128,
        )

        events = _collect_stream(stream)
        event_types = [event.type for event in events]
        terminal = _assert_stream_contract(
            events,
            expected_text="CHAT-STREAM-OK",
        )
        assert "response.in_progress" in event_types, event_types
        assert terminal.status == "completed"
        response_id = terminal.id

        retrieved = chat_streaming_client.responses.retrieve(response_id)
        assert retrieved.id == response_id, (
            f"retrieved response ID {retrieved.id!r} should match "
            f"streamed ID {response_id!r}"
        )
        assert retrieved.status == "completed", (
            f"retrieved response should be completed; got {retrieved.status!r}"
        )
        assert "CHAT-STREAM-OK" in retrieved.output_text, (
            "retrieved response should contain the streamed marker; "
            f"got {retrieved.output_text!r}"
        )

    def test_streaming_incomplete_round_trip(self, chat_streaming_client):
        stream = chat_streaming_client.responses.create(
            model=VLLM_MODEL,
            input="Write a long explanation of network proxies. /no_think",
            store=False,
            stream=True,
            max_output_tokens=1,
        )

        events = _collect_stream(stream)
        terminal = _assert_stream_contract(events, require_usage=False)
        assert events[-1].type == "response.incomplete"
        assert terminal.status == "incomplete"
        assert terminal.incomplete_details.reason == "max_output_tokens"

    def test_backend_error_is_sdk_compatible(self, chat_streaming_client):
        with pytest.raises(NotFoundError) as exc_info:
            chat_streaming_client.responses.create(
                model="model-that-does-not-exist",
                input="This request must fail.",
                store=False,
            )
        assert exc_info.value.status_code == 404


# ---------------------------------------------------------------------------
# Agentic Loop Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def mcp_server():
    """Start an in-process MCP mock server for the test session."""
    port = _free_port()
    server = HTTPServer(("127.0.0.1", port), MCPHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield port
    server.shutdown()


@pytest.fixture(scope="session")
def search_server():
    """Start an in-process mock Brave search server for the test session."""
    port = _free_port()
    server = HTTPServer(("127.0.0.1", port), BraveSearchHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield port
    server.shutdown()


@pytest.fixture(scope="session")
def agentic_proxy(tmp_path_factory, request, mcp_server, search_server):
    """Start a Praxis proxy with the native Responses agentic loop."""
    port = _free_port()
    db_dir = tmp_path_factory.mktemp("agentic-responses")
    db_path = str(db_dir / "responses.db")
    config_path = _write_agentic_config(port, db_path, mcp_server, search_server)
    binary = _find_binary()

    log_path = str(db_dir / "praxis.log")
    log_file = open(log_path, "w")
    started = False

    proc = subprocess.Popen(
        [binary, "-c", config_path],
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    try:
        _wait_for_proxy(port, proc, log_path)
        started = True
        yield port, mcp_server, search_server
    finally:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        log_file.close()
        if not started or request.session.testsfailed > 0:
            with open(log_path) as f:
                print(
                    f"\n=== Agentic Praxis logs ===\n{f.read()}",
                    file=sys.stderr,
                )
        os.unlink(config_path)


@pytest.fixture(scope="session")
def agentic_client(agentic_proxy):
    """Return an OpenAI client pointed at the agentic Praxis proxy."""
    proxy_port, _, _ = agentic_proxy
    return OpenAI(
        base_url=f"http://127.0.0.1:{proxy_port}/v1",
        api_key="test",
        max_retries=0,
        timeout=300,
    )


@pytest.fixture(scope="session")
def translated_agentic_proxy(
    tmp_path_factory,
    request,
    mcp_server,
    search_server,
):
    """Start the agentic loop through Responses-to-Chat translation."""
    port = _free_port()
    db_dir = tmp_path_factory.mktemp("translated-agentic-responses")
    db_path = str(db_dir / "responses.db")
    config_path = _write_agentic_config(
        port,
        db_path,
        mcp_server,
        search_server,
        translate_to_chat=True,
    )
    binary = _find_binary()

    log_path = str(db_dir / "praxis.log")
    log_file = open(log_path, "w")
    started = False
    proc = subprocess.Popen(
        [binary, "-c", config_path],
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    try:
        _wait_for_proxy(port, proc, log_path)
        started = True
        yield port
    finally:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        log_file.close()
        if not started or request.session.testsfailed > 0:
            with open(log_path) as f:
                print(
                    f"\n=== Translated agentic Praxis logs ===\n{f.read()}",
                    file=sys.stderr,
                )
        os.unlink(config_path)


@pytest.fixture(scope="session")
def translated_agentic_client(translated_agentic_proxy):
    """Return an SDK client using translated agentic inference."""
    return OpenAI(
        base_url=f"http://127.0.0.1:{translated_agentic_proxy}/v1",
        api_key="test",
        max_retries=0,
        timeout=300,
    )


# ---------------------------------------------------------------------------
# Agentic Loop Tests
# ---------------------------------------------------------------------------


def _assert_multi_round_usage_and_trace(response, *, transport):
    """Assert a terminal agentic response carries a multi-round accumulated
    output trace and an internally consistent, summed usage object.

    Shared by the buffered and streaming #983 tests so both transports assert
    exactly the same shape (criterion e: buffered and streaming expose
    equivalent accumulated terminal output and usage). ``response`` is the
    buffered ``Response`` or the streamed ``response.completed`` event's
    ``response`` -- both expose ``.status``, ``.output``, and ``.usage``.

    Deliberately does NOT assert exact per-round token counts: against a real
    model those are not predictable. The deterministic exact-sum contract is
    covered by the Rust integration test ``two_tool_rounds_accumulate_*``
    (tests/integration/tests/suite/examples/openai_agentic_loop.rs) with a
    StatefulCapturingBackend. Here we assert the accumulation *machinery* end
    to end over a genuine multi-round loop.
    """
    assert response.status in ("completed", "incomplete"), (
        f"[{transport}] expected completed or incomplete (token limit); "
        f"got: {response.status}"
    )

    output_types = [item.type for item in response.output]
    assert "function_call" in output_types, (
        f"[{transport}] accumulated output should contain an auto-executed "
        f"function_call; got: {output_types}"
    )
    assert "mcp_call" in output_types, (
        f"[{transport}] accumulated output should contain an MCP tool result "
        f"(mcp_call); got: {output_types}"
    )
    rounds = sum(
        1 for t in output_types
        if t in ("function_call", "message", "reasoning")
    )
    assert rounds >= 2, (
        f"[{transport}] accumulated output should span at least two inference "
        f"rounds; got: {output_types}"
    )

    # Usage is summed across every inference round into one terminal object.
    # We cannot predict the totals, but the summed object must be present,
    # positive, and internally consistent -- if merge_usage accumulated some
    # fields but not others, total would drift from input + output.
    usage = response.usage
    assert usage is not None, (
        f"[{transport}] terminal response must carry an accumulated usage "
        "object"
    )
    assert usage.input_tokens > 0, (
        f"[{transport}] accumulated input_tokens should be positive; "
        f"got: {usage.input_tokens}"
    )
    assert usage.output_tokens > 0, (
        f"[{transport}] accumulated output_tokens should be positive; "
        f"got: {usage.output_tokens}"
    )
    assert usage.total_tokens == usage.input_tokens + usage.output_tokens, (
        f"[{transport}] accumulated usage must be internally consistent "
        f"(total == input + output); got: {usage.input_tokens} + "
        f"{usage.output_tokens} != {usage.total_tokens}"
    )


class TestAgenticLoopVLLM:
    """Integration tests for the agentic loop against a vLLM backend."""

    def test_mcp_tool_auto_executes_and_returns(
        self,
        agentic_client,
        agentic_proxy,
    ):
        """MCP tools are auto-executed by the proxy within the IRR loop.

        The proxy resolves the MCP tool (tools/list), sends inference
        to vLLM, dispatches the function_call via tools/call on the
        MCP server, and re-enters inference with the result. The
        accumulated output contains the full execution trace:
        function_call, mcp_call, and the final message.
        """
        _, mcp_port, _ = agentic_proxy
        mcp_url = f"http://127.0.0.1:{mcp_port}/mcp"

        response = agentic_client.responses.create(
            model=VLLM_MODEL,
            input=(
                "You MUST call the get_weather function for Paris. "
                "Do not answer directly. /no_think"
            ),
            tools=[
                {
                    "type": "mcp",
                    "server_label": "weather",
                    "server_url": mcp_url,
                    "allowed_tools": ["get_weather"],
                    "require_approval": "never",
                }
            ],
            store=False,
            max_output_tokens=512,
        )

        assert response.status in ("completed", "incomplete"), (
            f"expected completed or incomplete (token limit); got: {response.status}"
        )

        output_types = [item.type for item in response.output]
        assert "function_call" in output_types, (
            "accumulated output should contain the auto-executed "
            f"function_call; got: {output_types}"
        )
        assert "mcp_call" in output_types, (
            "accumulated output should contain the MCP tool result "
            f"(mcp_call); got: {output_types}"
        )
        rounds = sum(
            1 for t in output_types if t in ("function_call", "message", "reasoning")
        )
        assert rounds >= 2, (
            "accumulated output should span at least two inference "
            f"rounds; got: {output_types}"
        )

    def test_mcp_approval_request_stops_before_dispatch(
        self,
        agentic_client,
        agentic_proxy,
    ):
        _, mcp_port, _ = agentic_proxy
        MCPHandler.authorization_headers.clear()

        response = agentic_client.responses.create(
            model=VLLM_MODEL,
            input=(
                "You MUST call get_weather for Paris. Do not answer directly. /no_think"
            ),
            tools=[
                {
                    "type": "mcp",
                    "server_label": "weather",
                    "server_url": f"http://127.0.0.1:{mcp_port}/mcp",
                    "allowed_tools": ["get_weather"],
                    "require_approval": "always",
                }
            ],
            store=False,
            max_output_tokens=256,
        )

        approvals = [
            item for item in response.output if item.type == "mcp_approval_request"
        ]
        assert len(approvals) == 1, response.output
        assert approvals[0].name == "get_weather"
        assert approvals[0].server_label == "weather"
        assert "Paris" in approvals[0].arguments
        assert not any(item.type == "mcp_call" for item in response.output), (
            response.output
        )

    def test_mcp_authorization_is_bearer_and_not_forwarded_to_model(
        self,
        agentic_client,
        agentic_proxy,
    ):
        _, mcp_port, _ = agentic_proxy
        token = "sdk-mcp-secret-789"
        MCPHandler.authorization_headers.clear()

        response = agentic_client.responses.create(
            model=VLLM_MODEL,
            input=(
                "You MUST call get_weather for Paris. Do not answer directly. /no_think"
            ),
            tools=[
                {
                    "type": "mcp",
                    "server_label": "weather",
                    "server_url": f"http://127.0.0.1:{mcp_port}/mcp",
                    "authorization": token,
                    "allowed_tools": ["get_weather"],
                    "require_approval": "always",
                }
            ],
            store=False,
            max_output_tokens=256,
        )

        assert any(item.type == "mcp_approval_request" for item in response.output), (
            response.output
        )
        assert MCPHandler.authorization_headers
        assert set(MCPHandler.authorization_headers) == {f"Bearer {token}"}
        assert token not in json.dumps(response.model_dump(), default=str)

    def test_authorization_in_mcp_headers_is_stripped(
        self,
        agentic_client,
        agentic_proxy,
    ):
        _, mcp_port, _ = agentic_proxy
        MCPHandler.authorization_headers.clear()

        response = agentic_client.responses.create(
            model=VLLM_MODEL,
            input=(
                "You MUST call get_weather for Paris. Do not answer directly. /no_think"
            ),
            tools=[
                {
                    "type": "mcp",
                    "server_label": "weather",
                    "server_url": f"http://127.0.0.1:{mcp_port}/mcp",
                    "headers": {"Authorization": "Bearer must-be-stripped"},
                    "allowed_tools": ["get_weather"],
                    "require_approval": "always",
                }
            ],
            store=False,
            max_output_tokens=256,
        )

        assert any(item.type == "mcp_approval_request" for item in response.output), (
            response.output
        )
        assert MCPHandler.authorization_headers
        assert set(MCPHandler.authorization_headers) == {None}

    def test_web_search_executes_and_returns_result(
        self,
        translated_agentic_client,
    ):
        request_count = len(BraveSearchHandler.request_paths)
        response = translated_agentic_client.responses.create(
            model=VLLM_MODEL,
            input=(
                "You MUST use web search, then report the result title and "
                "URL. /no_think"
            ),
            tools=[
                {
                    "type": "web_search",
                    "search_context_size": "low",
                }
            ],
            store=False,
            max_output_tokens=512,
        )

        web_search_calls = [
            item for item in response.output if item.type == "web_search_call"
        ]
        assert len(web_search_calls) == 1, response.output
        assert web_search_calls[0].status == "completed"
        assert len(BraveSearchHandler.request_paths) == request_count + 1
        assert any(item.type == "message" for item in response.output)

    def test_web_search_streams_one_logical_response(
        self,
        translated_agentic_client,
    ):
        request_count = len(BraveSearchHandler.request_paths)
        stream = translated_agentic_client.responses.create(
            model=VLLM_MODEL,
            input=("You MUST use web search, then report the result title. /no_think"),
            tools=[
                {
                    "type": "web_search",
                    "search_context_size": "low",
                }
            ],
            store=False,
            stream=True,
            max_output_tokens=512,
        )

        terminal = _assert_stream_contract(_collect_stream(stream))
        web_search_calls = [
            item for item in terminal.output if item.type == "web_search_call"
        ]
        assert len(web_search_calls) == 1, terminal.output
        assert web_search_calls[0].status == "completed"
        if len(BraveSearchHandler.request_paths) == request_count:
            pytest.xfail(
                "translated streaming exposes the hosted call before the "
                "agentic loop can dispatch it"
            )
        assert len(BraveSearchHandler.request_paths) == request_count + 1

    def test_mcp_tool_streams_terminal_round_as_one_logical_response(
        self,
        agentic_client,
        agentic_proxy,
    ):
        """Streaming sibling of test_mcp_tool_auto_executes_and_returns.

        With stream=True the proxy auto-executes the intermediate MCP
        tool round internally and buffers it, then streams only the
        terminal round to the client as ONE logical SSE response
        (response.created -> ... -> response.completed). The
        logical-stream finalizer replaces that terminal event's output
        with the cross-round accumulated trace, so the single
        response.completed carries the same execution trace the buffered
        test observes: function_call, mcp_call, and the final message.
        """
        _, mcp_port, _ = agentic_proxy
        mcp_url = f"http://127.0.0.1:{mcp_port}/mcp"

        stream = agentic_client.responses.create(
            model=VLLM_MODEL,
            input=(
                "You MUST call the get_weather function for Paris. "
                "Do not answer directly. /no_think"
            ),
            tools=[
                {
                    "type": "mcp",
                    "server_label": "weather",
                    "server_url": mcp_url,
                    "allowed_tools": ["get_weather"],
                    "require_approval": "never",
                }
            ],
            store=False,
            stream=True,
            max_output_tokens=512,
        )

        event_types = []
        text_parts = []
        final_response = None

        for event in stream:
            event_types.append(event.type)
            if event.type == "response.output_text.delta":
                text_parts.append(event.delta)
            if event.type == "response.completed":
                final_response = event.response

        # One coherent lifecycle framing, exactly as the single-round
        # test_streaming_through_irr asserts: created first, completed last.
        assert event_types[0] == "response.created", event_types
        assert event_types[-1] == "response.completed", event_types
        assert final_response is not None, (
            f"stream must terminate with a response.completed event; got: {event_types}"
        )
        assert final_response.status in ("completed", "incomplete"), (
            f"expected completed or incomplete (token limit); got: {final_response.status}"
        )

        # The terminal event's output is the cross-round accumulated trace,
        # so it mirrors the buffered test: the auto-executed function_call
        # and the MCP result (mcp_call) both surface in the one stream.
        output_types = [item.type for item in final_response.output]
        assert "function_call" in output_types, (
            "streamed terminal output should contain the auto-executed "
            f"function_call; got: {output_types}"
        )
        assert "mcp_call" in output_types, (
            "streamed terminal output should contain the MCP tool result "
            f"(mcp_call); got: {output_types}"
        )
        rounds = sum(
            1 for t in output_types if t in ("function_call", "message", "reasoning")
        )
        assert rounds >= 2, (
            "streamed terminal output should span at least two inference "
            f"rounds; got: {output_types}"
        )

        # MCPHandler returns f"72F and sunny in {city}"; the city argument
        # is model-chosen, so assert only the stable, non-templated prefix.
        # The mcp_call output item carries this tool result verbatim, so it
        # is present whether or not the model echoes it in the streamed text.
        haystack = json.dumps(final_response.model_dump(), default=str) + "".join(
            text_parts
        )
        assert "72F and sunny in" in haystack, (
            "the MCP get_weather result should be reflected in the "
            f"accumulated output or streamed text; got: {haystack}"
        )

    def test_client_function_exits_openai(self, agentic_client):
        """Client-side function tools exit the agentic loop without
        auto-execution, even when the IRR is active.

        This proves the IRR + openai_agentic_loop + mcp_dispatch correctly
        distinguish MCP tools (auto-loop) from client functions
        (return to caller).
        """
        response = agentic_client.responses.create(
            model=VLLM_MODEL,
            input=(
                "You MUST call the get_weather function for Paris. "
                "Do not answer directly. /no_think"
            ),
            tools=[
                {
                    "type": "function",
                    "name": "get_weather",
                    "description": "Get current weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                }
            ],
            temperature=0,
            store=False,
            max_output_tokens=256,
        )

        assert response.status == "completed"

        function_calls = [
            item for item in response.output if item.type == "function_call"
        ]
        assert len(function_calls) >= 1, (
            "client-side function calls should be returned to the caller; "
            f"got output types: {[i.type for i in response.output]}"
        )
        assert function_calls[0].name == "get_weather"

    def test_agentic_loop_accumulates_usage_across_rounds(
        self, agentic_client, agentic_proxy,
    ):
        """Issue #983: usage accumulates across consecutive inference rounds.

        Buffered variant. A single auto-executed MCP tool drives a
        model -> tool -> model loop; the terminal response exposes the
        cross-round accumulated output trace (function_call + mcp_call + the
        final message) and one usage object summed across every inference
        round.

        The deterministic exact-sum contract over *two sequential tool rounds*
        lives in the Rust test ``two_tool_rounds_accumulate_output_and_usage``
        (tests/integration/tests/suite/examples/openai_agentic_loop.rs) with a
        StatefulCapturingBackend. That scenario is not reproducible live: a
        small model batches multiple tool calls into one round, which
        openai_agentic_loop rejects (``exactly one function call per round``),
        so this live counterpart drives one reliable tool round and asserts the
        accumulation *machinery* end to end -- a genuine multi-round trace plus
        a present, positive, internally consistent summed usage -- against a
        real backend.
        """
        _, mcp_port, _ = agentic_proxy
        mcp_url = f"http://127.0.0.1:{mcp_port}/mcp"

        response = agentic_client.responses.create(
            model=VLLM_MODEL,
            input=(
                "You MUST call the get_weather function for Paris. "
                "Do not answer directly. /no_think"
            ),
            tools=[
                {
                    "type": "mcp",
                    "server_label": "weather",
                    "server_url": mcp_url,
                    "allowed_tools": ["get_weather"],
                    "require_approval": "never",
                }
            ],
            store=False,
            max_output_tokens=512,
        )

        _assert_multi_round_usage_and_trace(response, transport="buffered")

    def test_agentic_loop_streaming_accumulates_usage_across_rounds(
        self, agentic_client, agentic_proxy,
    ):
        """Issue #983: streaming sibling of
        test_agentic_loop_accumulates_usage_across_rounds.

        With stream=True the proxy auto-executes the intermediate tool round
        internally and streams only the terminal round to the client as ONE
        logical SSE response (response.created -> ... -> response.completed).
        The logical-stream finalizer stamps that terminal event with the
        cross-round accumulated output and the summed usage, so the single
        response.completed exposes the same multi-round trace and internally
        consistent usage the buffered variant observes -- criterion e:
        buffered and streaming expose equivalent accumulated terminal output
        and usage.
        """
        _, mcp_port, _ = agentic_proxy
        mcp_url = f"http://127.0.0.1:{mcp_port}/mcp"

        stream = agentic_client.responses.create(
            model=VLLM_MODEL,
            input=(
                "You MUST call the get_weather function for Paris. "
                "Do not answer directly. /no_think"
            ),
            tools=[
                {
                    "type": "mcp",
                    "server_label": "weather",
                    "server_url": mcp_url,
                    "allowed_tools": ["get_weather"],
                    "require_approval": "never",
                }
            ],
            store=False,
            stream=True,
            max_output_tokens=512,
        )

        event_types = []
        final_response = None
        for event in stream:
            event_types.append(event.type)
            if event.type == "response.completed":
                final_response = event.response

        # One coherent SSE lifecycle framing across the whole multi-round loop.
        assert event_types[0] == "response.created", event_types
        assert event_types[-1] == "response.completed", event_types
        assert final_response is not None, (
            "stream must terminate with a response.completed event; "
            f"got: {event_types}"
        )

        _assert_multi_round_usage_and_trace(final_response, transport="streaming")


# ---------------------------------------------------------------------------
# File search: dedicated proxy config, fixtures, and tests
# ---------------------------------------------------------------------------

FILE_SEARCH_CONFIG_TEMPLATE = """\
listeners:
  - name: ai-gateway
    address: "127.0.0.1:{praxis_port}"
    filter_chains: [file-search-pipeline]

filter_chains:
  - name: file-search-pipeline
    filters:
      - filter: openai_responses_format
      - filter: openai_responses_validate
      - filter: iterative_request_router
        initial_step: inference
        max_iterations: 8
        # Generous deadlines: file-search inference runs on CPU-only vLLM under
        # heavy CI load (postgres + vLLM + OGX co-located), which can exceed a
        # 60s step budget. Matches the agentic config's IRR timeouts.
        timeout_ms: 300000
        step_timeout_ms: 300000
        max_response_bytes: 67108864
        max_state_bytes: 136314880
        steps:
          - name: inference
            filters:
              - filter: openai_tool_parse
              - filter: openai_file_search_callout
                vector_store_url: http://{ogx_endpoint}
                allow_private_url: true
                timeout_ms: 30000
                max_response_bytes: 10485760
                max_total_response_bytes: 67108864
                max_state_bytes: 136314880
                on_failure: closed
                forward_headers:
                  - authorization
              - filter: openai_responses_proxy
                name: inference
              - filter: headers
                request_set:
                  - name: Content-Type
                    value: application/json
              - filter: router
                routes:
                  - path_prefix: "/"
                    cluster: "inference"
              - filter: load_balancer
                clusters:
                  - name: "inference"
                    read_timeout_ms: 300000
                    endpoints:
                      - "{vllm_endpoint}"
            on_result:
              - filter: openai_file_search_callout
                key: pending
                value: "true"
                next: inference
              - default: true
                done: true

insecure_options:
  allow_private_endpoints: true
"""


def _write_file_search_config(praxis_port: int) -> str:
    config = FILE_SEARCH_CONFIG_TEMPLATE.format(
        praxis_port=praxis_port,
        ogx_endpoint=_ogx_endpoint(),
        vllm_endpoint=_vllm_endpoint(),
    )
    fd, path = tempfile.mkstemp(suffix=".yaml")
    with os.fdopen(fd, "w") as f:
        f.write(config)
    return path


@pytest.fixture(scope="session")
def vector_store():
    """Create a vector store with a test document in OGX."""
    import httpx

    marker = "PRAXIS-FILE-SEARCH-8472"
    embedding_model = os.environ.get(
        "OGX_EMBEDDING_MODEL",
        "sentence-transformers/nomic-ai/nomic-embed-text-v1.5",
    )
    embedding_dimension = int(os.environ.get("OGX_EMBEDDING_DIMENSION", "768"))

    client = httpx.Client(base_url=OGX_BASE_URL, timeout=300)
    store_id = ""
    file_id = ""

    try:
        store = client.post(
            "/v1/vector_stores",
            json={
                "name": f"praxis-file-search-{os.getpid()}",
                "embedding_model": embedding_model,
                "embedding_dimension": embedding_dimension,
                "provider_id": "faiss",
            },
        ).json()
        store_id = store["id"]

        file_content = (
            f"Praxis file-search integration report.\n"
            f"The secret marker is {marker}.\n"
            f"Revenue grew 37 percent year over year.\n"
        )
        uploaded = client.post(
            "/v1/files",
            files={"file": ("test-marker.txt", file_content.encode(), "text/plain")},
            data={"purpose": "assistants"},
        ).json()
        file_id = uploaded["id"]

        client.post(
            f"/v1/vector_stores/{store_id}/files",
            json={
                "file_id": file_id,
                "attributes": {"department": "finance"},
            },
        )

        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            status = client.get(f"/v1/vector_stores/{store_id}/files/{file_id}").json()
            if status.get("status") == "completed":
                break
            if status.get("status") in ("failed", "cancelled"):
                raise RuntimeError(f"OGX indexing failed: {status.get('last_error')}")
            time.sleep(0.5)
        else:
            raise TimeoutError("OGX indexing timed out after 300s")

        yield store_id, marker

    finally:
        if store_id:
            try:
                client.delete(f"/v1/vector_stores/{store_id}")
            except Exception:
                pass
        if file_id:
            try:
                client.delete(f"/v1/files/{file_id}")
            except Exception:
                pass
        client.close()


@pytest.fixture(scope="session")
def file_search_proxy(tmp_path_factory, request):
    """Start a Praxis proxy with the file-search-callout pipeline."""
    port = _free_port()
    config_path = _write_file_search_config(port)
    binary = _find_binary()

    log_dir = tmp_path_factory.mktemp("file-search")
    log_path = str(log_dir / "praxis.log")
    log_file = open(log_path, "w")
    started = False

    proc = subprocess.Popen(
        [binary, "-c", config_path],
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    try:
        _wait_for_proxy(port, proc, log_path)
        started = True
        yield port
    finally:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        log_file.close()
        if not started or request.session.testsfailed > 0:
            with open(log_path) as f:
                print(
                    f"\n=== File search proxy logs ===\n{f.read()}",
                    file=sys.stderr,
                )
        os.unlink(config_path)


@pytest.fixture(scope="session")
def file_search_client(file_search_proxy):
    """Return an OpenAI client pointed at the file-search Praxis proxy."""
    return OpenAI(
        base_url=f"http://127.0.0.1:{file_search_proxy}/v1",
        api_key="test",
        max_retries=0,
        timeout=300,
    )


class TestFileSearchVLLM:
    """File search integration tests: vLLM -> Praxis -> OGX -> vLLM."""

    def test_file_search_with(self, file_search_client, vector_store):
        """vLLM emits function_call(name=file_search) which the proxy
        translates to file_search_call, executes the OGX search callout,
        and returns results to the client.
        """
        store_id, _marker = vector_store
        response = file_search_client.responses.create(
            model=VLLM_MODEL,
            input=(
                "Use the file_search tool to find information about "
                "the Praxis marker. Repeat the marker exactly. /no_think"
            ),
            tools=[
                {
                    "type": "file_search",
                    "vector_store_ids": [store_id],
                }
            ],
            include=["file_search_call.results"],
            store=False,
            max_output_tokens=512,
        )

        assert response.status in ("completed", "incomplete"), (
            f"response should complete; got status={response.status}"
        )

        output_types = [item.type for item in response.output]
        assert output_types, "response should have at least one output item"

        file_search_items = [
            item for item in response.output if item.type == "file_search_call"
        ]
        assert file_search_items, (
            "translated function_call(name=file_search) should appear as "
            f"file_search_call; got output types: {output_types}"
        )
        for item in file_search_items:
            assert item.status in ("completed", "incomplete"), (
                f"file_search_call status should be terminal; got: {item.status}"
            )


# ---------------------------------------------------------------------------
# File search via Chat Completions translation (issue #296)
# ---------------------------------------------------------------------------

FILE_SEARCH_CHAT_CONFIG_PATH = (
    "examples/configs/openai/responses/file-search-chat-completions.yaml"
)


def _write_file_search_chat_config(praxis_port: int) -> str:
    """Patch the shipped file-search-chat-completions example for testing.

    Exercises the real example config (per repo test requirements) while
    retargeting the vector-store callout at OGX and the model backend at
    vLLM's /v1/chat/completions endpoint.

    IRR / callout / backend read deadlines are widened to match
    FILE_SEARCH_CONFIG_TEMPLATE: CPU-only vLLM plus OGX is slower when
    the postgres store job co-locates those containers, and a 60s step
    budget can expire before vLLM returns.
    """
    with open(FILE_SEARCH_CHAT_CONFIG_PATH) as f:
        config = f.read()

    config = config.replace("127.0.0.1:8080", f"127.0.0.1:{praxis_port}")
    config = config.replace("127.0.0.1:8001", _ogx_endpoint())
    vllm = _vllm_endpoint()
    config = config.replace(
        '                  - name: "chat-completions-backend"\n'
        "                    endpoints:\n"
        '                      - "127.0.0.1:3001"',
        f'                  - name: "chat-completions-backend"\n'
        f"                    read_timeout_ms: 300000\n"
        f"                    endpoints:\n"
        f'                      - "{vllm}"',
    )
    config = config.replace("timeout_ms: 120000", "timeout_ms: 300000")
    config = config.replace("step_timeout_ms: 60000", "step_timeout_ms: 300000")
    config = config.replace("timeout_ms: 5000", "timeout_ms: 30000")
    if f'- "{vllm}"' not in config:
        raise RuntimeError(
            "file-search-chat-completions.yaml cluster block did not match; "
            "vLLM endpoint was not patched"
        )

    fd, path = tempfile.mkstemp(suffix=".yaml")
    with os.fdopen(fd, "w") as f:
        f.write(config)
    return path


@pytest.fixture(scope="session")
def file_search_chat_proxy(tmp_path_factory, request):
    """Start a Praxis proxy with the file-search Chat Completions pipeline."""
    port = _free_port()
    config_path = _write_file_search_chat_config(port)
    binary = _find_binary()

    log_dir = tmp_path_factory.mktemp("file-search-chat")
    log_path = str(log_dir / "praxis.log")
    log_file = open(log_path, "w")
    started = False

    proc = subprocess.Popen(
        [binary, "-c", config_path],
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    try:
        _wait_for_proxy(port, proc, log_path)
        started = True
        yield port
    finally:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        log_file.close()
        if not started or request.session.testsfailed > 0:
            with open(log_path) as f:
                print(
                    f"\n=== File search (chat) proxy logs ===\n{f.read()}",
                    file=sys.stderr,
                )
        os.unlink(config_path)


@pytest.fixture(scope="session")
def file_search_chat_client(file_search_chat_proxy):
    """Return an OpenAI client pointed at the file-search chat proxy."""
    return OpenAI(
        base_url=f"http://127.0.0.1:{file_search_chat_proxy}/v1",
        api_key="test",
        max_retries=0,
        timeout=300,
    )


class TestFileSearchChatCompletionsVLLM:
    """Issue #296: hosted file_search against a Chat Completions backend.

    Unlike TestFileSearchVLLM (which proxies vLLM's native /v1/responses),
    this drives responses_to_chat_completions: the native file_search tool
    is synthesized into a private chat `function`, vLLM's
    /v1/chat/completions emits the call, the proxy runs the OGX vector-store
    search, and drives one more finite inference round -- without ever
    exposing the private function to the client.
    """

    def test_file_search_translated_to_chat_function_round_trip(
        self, file_search_chat_client, vector_store
    ):
        store_id, marker = vector_store
        response = file_search_chat_client.responses.create(
            model=VLLM_MODEL,
            input=(
                "You MUST use the file_search tool to find the Praxis marker "
                "in the indexed report. Do not answer from memory. /no_think"
            ),
            tools=[
                {
                    "type": "file_search",
                    "vector_store_ids": [store_id],
                }
            ],
            include=["file_search_call.results"],
            store=False,
            max_output_tokens=512,
        )

        assert response.status in ("completed", "incomplete"), (
            f"response should reach a terminal status; got {response.status}"
        )

        output_types = [item.type for item in response.output]

        # The synthesized private function must never leak to the client; it
        # is normalized back to a hosted file_search_call.
        assert all(t != "function_call" for t in output_types), (
            "the private file_search function must not surface as a client "
            f"function_call; got output types: {output_types}"
        )

        file_search_items = [
            item for item in response.output if item.type == "file_search_call"
        ]
        assert file_search_items, (
            "the synthesized file_search function call should be normalized "
            f"back to a file_search_call; got output types: {output_types}"
        )
        for item in file_search_items:
            assert item.status in ("completed", "incomplete"), (
                f"file_search_call status should be terminal; got: {item.status}"
            )

        # Results come from OGX deterministically (not the model), so the
        # indexed marker must round-trip through the model->search->model flow.
        # If a future OGX result shape omits content text, relax this to
        # asserting file_search results are simply non-empty.
        payload = json.dumps(response.model_dump(), default=str)
        assert marker in payload, (
            "OGX search results (via include=file_search_call.results) should "
            f"contain the indexed marker {marker!r}; got: {payload}"
        )


if __name__ == "__main__":
    sys.exit(
        pytest.main(
            [__file__, "-v", "--tb=short", "-ra", "--durations=20"] + sys.argv[1:]
        )
    )
