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
OpenAI Responses API integration tests against either a lightweight simulator
or a real vLLM backend.

The default ``VLLM_TEST_BACKEND=live`` mode starts Praxis against real vLLM.
``VLLM_TEST_BACKEND=simulator`` uses llm-d-inference-sim for deterministic
gateway, persistence, tool-loop, and SDK protocol coverage. Tests marked
``real_inference`` or ``vllm_compat`` are skipped in simulator mode.

Usage:
    cargo build -p praxis-ai-proxy --features full,store-sqlite
    uv run tests/integration/sdk/openai/test_openai_responses_vllm.py -s
"""

import base64
from concurrent.futures import ThreadPoolExecutor
import io
import json
import os
import signal
import socket
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
from typing import Any, ClassVar
from urllib.parse import urlparse

import httpx
import pytest
from openai import (
    APIConnectionError,
    APIStatusError,
    BadRequestError,
    NotFoundError,
    OpenAI,
    PermissionDeniedError,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen3-0.6B")
VLLM_TEST_BACKEND = os.environ.get("VLLM_TEST_BACKEND", "live")
OGX_BASE_URL = os.environ.get("OGX_BASE_URL", "http://127.0.0.1:8321")
PRAXIS_AI_BIN = os.environ.get("PRAXIS_AI_BIN")
DATABASE_URL = os.environ.get("DATABASE_URL", "")
REQUIRE_LIVE_WEB_SEARCH = os.environ.get("PRAXIS_TEST_REQUIRE_LIVE_WEB_SEARCH") == "1"
CONFIG_PATH = "examples/configs/openai/responses/full-flow-agentic.yaml"
AGENTIC_CONFIG_PATH = "examples/configs/openai/responses/agentic-loop.yaml"
IRR_STREAMING_CONFIG_PATH = (
    "examples/configs/openai/responses/irr-terminal-streaming.yaml"
)
CHAT_STREAMING_CONFIG_PATH = (
    "examples/configs/openai/responses/responses-to-chat-completions.yaml"
)
REASONING_CONFIG_PATH = (
    "examples/configs/openai/responses/responses-to-chat-completions-reasoning.yaml"
)
COMPACT_CONFIG_PATH = "examples/configs/openai/responses/compact.yaml"
WEB_SEARCH_CHAT_STREAMING_CONFIG_PATH = (
    "examples/configs/openai/responses/web-search-chat-completions.yaml"
)
# The full-flow example trusts these only after an authentication gateway has
# overwritten them. This harness connects directly to Praxis, so it emulates
# that boundary for clients using the full-flow configuration.
TRUSTED_OWNER_HEADERS = {
    "x-auth-tenant": "test-tenant",
    "x-auth-user": "test-user",
}
CLIENT_TOOL_COMPAT_CONFIG_PATH = (
    "examples/configs/openai/responses/client-tool-compat.yaml"
)
CLIENT_TOOL_COMPAT_CHAT_CONFIG_PATH = (
    "examples/configs/openai/responses/client-tool-compat-chat-completions.yaml"
)

TERMINAL_RESPONSE_EVENTS = {
    "response.cancelled",
    "response.completed",
    "response.failed",
    "response.incomplete",
}

if VLLM_TEST_BACKEND not in {"live", "simulator"}:
    raise RuntimeError(
        "VLLM_TEST_BACKEND must be either 'live' or 'simulator'; "
        f"got {VLLM_TEST_BACKEND!r}"
    )


def requires_real_inference(test):
    """Mark a semantic inference test and skip it on the simulator."""
    test = pytest.mark.real_inference(test)
    return pytest.mark.skipif(
        VLLM_TEST_BACKEND != "live",
        reason="test requires real model inference",
    )(test)


def requires_vllm_compat(test):
    """Mark a vLLM-specific contract test and skip it on the simulator."""
    test = pytest.mark.vllm_compat(test)
    return pytest.mark.skipif(
        VLLM_TEST_BACKEND != "live",
        reason="test requires real vLLM compatibility behavior",
    )(test)


def qualification_profile(name):
    """Override GPU report attribution for a test with dynamic fixtures."""
    def decorate(test):
        test.qualification_profile = name
        return test
    return decorate

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
        "praxis-ai binary not found: run `cargo build -p praxis-ai-proxy --features full` first"
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


def _make_openai_client(
    praxis_port: int,
    *,
    default_headers: dict[str, str] | None = None,
    timeout: float = 300,
) -> OpenAI:
    """Point an SDK client at a test proxy with the shared client settings."""
    options: dict[str, Any] = {}
    if default_headers is not None:
        options["default_headers"] = default_headers
    return OpenAI(
        base_url=f"http://127.0.0.1:{praxis_port}/v1",
        api_key="test",
        max_retries=0,
        timeout=timeout,
        **options,
    )


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


def _enable_response_store_compression(config: str) -> str:
    """Append a zstd compression block to the openai_response_store filter."""
    anchor = (
        "        responses_table: openai_responses\n"
        "        conversations_table: openai_conversations\n"
    )
    if anchor not in config:
        raise AssertionError(
            "response-store filter anchor not found; the example config layout "
            "changed and _enable_response_store_compression needs updating"
        )
    return config.replace(
        anchor,
        anchor + "        compression:\n          algorithm: zstd\n          level: 3\n",
    )


def _persist_config(config: str) -> str:
    """Write a generated Praxis config to a temp file and return its path.

    When the harness runs as root — as it does on the ephemeral EC2 GPU runner
    used by the nightly/label-triggered full suite — Praxis refuses to start
    unless ``insecure_options.allow_root`` is set. Inject it here so every config
    writer inherits the override in one place; non-root local and CPU CI runs are
    left byte-for-byte unchanged.
    """
    if os.geteuid() == 0 and "allow_root:" not in config:
        block = "\ninsecure_options:\n"
        override = "\ninsecure_options:\n  allow_root: true\n"
        if block in config:
            config = config.replace(block, override, 1)
        elif config.startswith("insecure_options:\n"):
            config = "insecure_options:\n  allow_root: true\n" + config[len("insecure_options:\n") :]
        else:
            config = config.rstrip("\n") + override

    fd, path = tempfile.mkstemp(suffix=".yaml")
    with os.fdopen(fd, "w") as handle:
        handle.write(config)
    return path


def _load_example_config(
    example_path: str,
    praxis_port: int,
    *,
    backend_endpoint: str | None = None,
    db_path: str | None = None,
) -> str:
    """Load a shipped example and substitute its routine test endpoints."""
    with open(example_path) as f:
        config = f.read()

    config = config.replace("127.0.0.1:8080", f"127.0.0.1:{praxis_port}")
    if backend_endpoint is not None:
        config = config.replace("127.0.0.1:3001", backend_endpoint)
    if db_path is not None:
        config = _patch_store_backend(config, db_path)
    return config


def _write_example_config(
    example_path: str,
    praxis_port: int,
    *,
    backend_endpoint: str | None = None,
    db_path: str | None = None,
) -> str:
    return _persist_config(
        _load_example_config(
            example_path,
            praxis_port,
            backend_endpoint=backend_endpoint,
            db_path=db_path,
        )
    )


def _write_full_flow_config(
    praxis_port: int,
    db_path: str,
    compression: bool = False,
    *,
    backend_endpoint: str | None = None,
    search_port: int | None = None,
    max_event_bytes: int | None = None,
    compact_callout_port: int | None = None,
    retained_limit: int | None = None,
    skip_agentic_response: bool = False,
) -> str:
    """Patch the shared full-flow example for live or recording backends."""
    config = _load_example_config(
        CONFIG_PATH,
        praxis_port,
        backend_endpoint=backend_endpoint or _vllm_endpoint(),
    )
    config = config.replace("127.0.0.1:9999", _ogx_endpoint())
    # The unified gateway wires openai_web_search into the IRR; its config
    # resolves ${WEB_SEARCH_API_KEY} at startup and fails closed when unset.
    # These vLLM turns never emit a web_search_call, so a literal placeholder
    # key keeps the dispatcher inert while letting the binary start.
    config = config.replace("api_key: ${WEB_SEARCH_API_KEY}", "api_key: test-key")
    if search_port is not None:
        search_anchor = "api_key: test-key\n                # Require the per-user key"
        assert config.count(search_anchor) == 1
        config = config.replace(
            search_anchor,
            "api_key: test-key\n"
            f"                base_url: http://127.0.0.1:{search_port}\n"
            "                # Require the per-user key",
        )
    if max_event_bytes is not None:
        anchor = (
            "        responses_table: openai_responses\n"
            "        conversations_table: openai_conversations\n"
        )
        assert config.count(anchor) == 1
        config = config.replace(anchor, anchor + f"        max_event_bytes: {max_event_bytes}\n")
    if compact_callout_port is not None:
        anchor = "      - filter: iterative_request_router\n"
        assert config.count(anchor) == 1
        config = config.replace(
            anchor,
            "      - filter: openai_responses_compact\n"
            "        allow_pre_security_callout: true\n"
            f"        inference_url: http://127.0.0.1:{compact_callout_port}/v1/chat/completions\n"
            "        allow_private_inference_url: true\n"
            f"        default_model: {VLLM_MODEL}\n"
            "        on_failure: open\n"
            "        conditions:\n"
            "          - unless:\n"
            "              bound_upstream:\n"
            "                application_provider: openai\n"
            + anchor,
        )
    if retained_limit is not None:
        anchor = "              - filter: openai_agentic_loop\n                max_infer_iters: 7\n"
        assert config.count(anchor) == 1
        config = config.replace(anchor, anchor + f"                max_retained_bytes: {retained_limit}\n")
    if skip_agentic_response:
        anchor = "              - filter: openai_agentic_loop\n                max_infer_iters: 7\n"
        assert config.count(anchor) == 1
        config = config.replace(
            anchor,
            anchor
            + "                response_conditions:\n"
            + "                  - when:\n"
            + "                      headers:\n"
            + "                        x-test-run-agentic: \"true\"\n",
        )
    if compression:
        config = _enable_response_store_compression(config)

    config = _patch_store_backend(config, db_path)
    return _persist_config(config)


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


def _write_reasoning_backend_config(
    praxis_port: int,
    db_path: str,
    backend_port: int,
    dialect: str = "vllm",
) -> str:
    """Patch the reasoning example to target a specific Chat backend port.

    The ``127.0.0.1:3001`` backend is pointed at ``backend_port`` (a capturing
    mock) so a test can observe the exact Chat Completions request body the
    backend receives after the proxy replays reasoning in the assistant
    reasoning field.
    """
    config = _load_example_config(
        REASONING_CONFIG_PATH,
        praxis_port,
        backend_endpoint=f"127.0.0.1:{backend_port}",
        db_path=db_path,
    )
    config = config.replace("dialect: vllm", f"dialect: {dialect}")
    return _persist_config(config)


@qualification_profile("supporting")
def test_reasoning_config_writers_inherit_root_override(tmp_path, monkeypatch):
    """Keep every reasoning fixture compatible with the root-run GPU worker."""
    monkeypatch.setattr(os, "geteuid", lambda: 0)
    db_path = str(tmp_path / "responses.db")
    paths = [
        _write_example_config(
            REASONING_CONFIG_PATH,
            18_080,
            backend_endpoint=_vllm_endpoint(),
            db_path=db_path,
        ),
        _write_reasoning_backend_config(18_081, db_path, 18_082),
    ]
    try:
        for path in paths:
            with open(path) as config_file:
                config = config_file.read()
            assert config.count("allow_root: true") == 1, config
    finally:
        for path in paths:
            os.unlink(path)


def _write_compact_config(
    praxis_port: int,
    db_path: str,
    compaction_port: int,
) -> str:
    """Patch the compact example for inference and a deterministic summary."""
    config = _load_example_config(COMPACT_CONFIG_PATH, praxis_port)
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
    return _persist_config(config)


@qualification_profile("supporting")
@pytest.mark.parametrize(
    ("writer", "args", "postgres_port"),
    [
        (_write_full_flow_config, (18_080, "/unused.db"), 9999),
        (_write_compact_config, (18_080, "/unused.db", 18_081), 9999),
        (_write_compact_config, (18_080, "/unused.db", 18_081), 11434),
    ],
)
def test_store_url_survives_endpoint_rewrites(writer, args, postgres_port, monkeypatch):
    database_url = f"postgres://test:test@127.0.0.1:{postgres_port}/responses"
    monkeypatch.setattr(sys.modules[__name__], "DATABASE_URL", database_url)
    path = writer(*args)
    try:
        with open(path) as config_file:
            config = config_file.read()
        assert f'database_url: "{database_url}"' in config
    finally:
        os.unlink(path)


def _write_web_search_chat_streaming_config(
    praxis_port: int, search_port: int, backend_endpoint: str
) -> str:
    """Patch the streaming web-search-through-Chat example for testing.

    Points the loop at the selected Chat backend and swaps the Brave provider's
    ``${WEB_SEARCH_API_KEY}`` placeholder for the in-process mock search server.
    """
    config = _load_example_config(WEB_SEARCH_CHAT_STREAMING_CONFIG_PATH, praxis_port)
    config = config.replace(
        '- "127.0.0.1:3001"',
        f'- "{backend_endpoint}"\n'
        "                    read_timeout_ms: 300000",
    )
    config = config.replace(
        "- filter: openai_web_search\n"
        "                provider: brave\n"
        "                api_key: ${WEB_SEARCH_API_KEY}",
        "- filter: openai_web_search\n"
        "                provider: brave\n"
        "                api_key: test-key\n"
        f"                base_url: http://127.0.0.1:{search_port}",
    )
    # The provider callout targets a loopback mock, so the executor's SSRF check
    # requires the operator opt-in on the outbound pipeline.
    config = config.replace(
        "allow_private_endpoints: true",
        "allow_private_endpoints: true\n  allow_private_upstreams: true",
    )

    path = _persist_config(config)
    return path


def _wait_for_proxy(
    port: int, proc: subprocess.Popen, log_path: str, timeout: float = 30.0
) -> None:
    """Wait for the listener and any asynchronous store provisioning."""
    deadline = time.monotonic() + timeout
    readiness_url = f"http://127.0.0.1:{port}/v1/responses/__praxis_readiness__"
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
                pass
        except OSError:
            time.sleep(0.2)
            continue

        try:
            response = httpx.get(
                readiness_url,
                headers=TRUSTED_OWNER_HEADERS,
                timeout=0.5,
            )
        except httpx.HTTPError:
            time.sleep(0.2)
            continue
        if not (
            response.status_code == 503
            and "Persisted state is still initializing." in response.text
        ):
            return
        time.sleep(0.2)
    raise TimeoutError(
        f"Praxis did not become ready within {timeout}s on port {port}:\n"
        f"{_read_log_tail(log_path)}"
    )


# ---------------------------------------------------------------------------
# MCP Mock Server
# ---------------------------------------------------------------------------


class MCPHandler(BaseHTTPRequestHandler):
    """Streamable HTTP MCP server with deterministic weather/time tools."""

    authorization_headers: ClassVar[list[str | None]] = []

    _tool_call_count = 0
    _tool_call_count_lock = threading.Lock()

    @classmethod
    def tool_call_count(cls) -> int:
        """Return the number of tool calls received by this test process."""
        with cls._tool_call_count_lock:
            return cls._tool_call_count

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
                        },
                        {
                            "name": "get_time",
                            "description": "Get the current local time for a city",
                            "inputSchema": {
                                "type": "object",
                                "properties": {"city": {"type": "string"}},
                                "required": ["city"],
                                "additionalProperties": False,
                            },
                        },
                        {
                            "name": "get_weather_map",
                            "description": "Get a weather map image link for a city",
                            "inputSchema": {
                                "type": "object",
                                "properties": {"city": {"type": "string"}},
                                "required": ["city"],
                                "additionalProperties": False,
                            },
                        },
                    ]
                },
            )
        elif method == "tools/call":
            with self._tool_call_count_lock:
                type(self)._tool_call_count += 1
            params = req.get("params", {})
            tool_name = params.get("name")
            city = params.get("arguments", {}).get("city", "unknown")
            if tool_name == "get_weather_map":
                # Non-text MCP content: a resource_link block. The tool result is
                # server-controlled and deterministic, so this exercises the
                # proxy's lossless non-text serialization end to end -- through
                # the wire and into the mcp_call output item the OpenAI SDK
                # deserializes -- independent of the model's choices.
                self._json_rpc(
                    rid,
                    {
                        "content": [
                            {
                                "type": "resource_link",
                                "uri": "file:///weather/paris-map.png",
                                "name": "paris-weather-map",
                                "mimeType": "image/png",
                            }
                        ]
                    },
                )
            else:
                result = (
                    f"12:00 PM in {city}"
                    if tool_name == "get_time"
                    else f"72F and sunny in {city}"
                )
                self._json_rpc(
                    rid, {"content": [{"type": "text", "text": result}]}
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
    """Mock Brave Search API returning canned results.

    Counts every dispatched query so tests can assert the web-search
    provider is invoked exactly once per agentic round.
    """

    #: Total queries served across all instances since the last reset.
    request_count = 0

    @classmethod
    def reset(cls):
        cls.request_count = 0

    request_paths: ClassVar[list[str]] = []

    def do_GET(self):
        type(self).request_paths.append(self.path)
        BraveSearchHandler.request_count += 1
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


class ChatCaptureHandler(BaseHTTPRequestHandler):
    """Capturing mock Chat Completions backend for reasoning-replay tests.

    Records each request body and returns a fixed, properly framed completion
    so a test can assert on exactly what the proxy forwards upstream without
    depending on live vLLM or model output.
    """

    captured_bodies: ClassVar[list[dict]] = []

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b""
        if body:
            try:
                type(self).captured_bodies.append(json.loads(body))
            except json.JSONDecodeError:
                pass
        payload = json.dumps(
            {
                "id": "chatcmpl_reasoning_capture",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": VLLM_MODEL,
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "reasoning": "I picked 42.",
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 1,
                    "total_tokens": 13,
                },
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt, *args):
        pass


class ResponsesWitnessHandler(BaseHTTPRequestHandler):
    """Recording shim that sits between Praxis and the inference backend.

    Captures the JSON body of every request the backend receives, then forwards
    it transparently and streams the response back so the full
    native Responses pipeline still completes. The synthetic
    ``sdk-conversation-stream`` model returns deterministic native responses
    for Conversation append tests. Tests use the captured bodies to assert on
    what the proxy actually forwards upstream after its rewrites.
    """

    forwarded_bodies: ClassVar[list[dict]] = []
    concurrent_barrier: ClassVar[threading.Barrier | None] = None
    terminal_gate: ClassVar[threading.Event | None] = None
    tool_round_count: ClassVar[int] = 0
    last_response_id: ClassVar[str | None] = None

    def log_message(self, fmt, *args):
        pass

    def _forward(self):
        """Capture the request and serve a synthetic or forwarded response."""
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b""
        request_body = None
        if body:
            try:
                request_body = json.loads(body)
                type(self).forwarded_bodies.append(request_body)
            except json.JSONDecodeError:
                pass
        if request_body and request_body.get("background") is True:
            self._send_background_response(request_body)
            return
        if request_body and request_body.get("prompt") is not None:
            self._send_prompt_response(request_body)
            return
        if request_body and request_body.get("model") in {
            "sdk-conversation-stream",
            "sdk-conversation-tool-stream",
        }:
            self._send_conversation_response(request_body)
            return
        headers = {
            k: v
            for k, v in self.headers.items()
            if k.lower() not in ("host", "content-length")
        }
        url = f"{VLLM_BASE_URL.rstrip('/')}{self.path}"
        with httpx.Client(timeout=300.0) as client:
            with client.stream(
                self.command, url, headers=headers, content=body
            ) as upstream:
                self.send_response(upstream.status_code)
                for key, value in upstream.headers.items():
                    if key.lower() in (
                        "transfer-encoding",
                        "content-length",
                        "connection",
                    ):
                        continue
                    self.send_header(key, value)
                self.end_headers()
                for chunk in upstream.iter_raw():
                    if chunk:
                        self.wfile.write(chunk)
                        self.wfile.flush()

    def _send_background_response(self, request_body):
        """Serve a provider-owned background create response."""
        response = {
            "id": f"resp_sdk_background_{time.time_ns()}",
            "object": "response",
            "created_at": int(time.time()),
            "model": request_body.get("model", "gpt-5"),
            "status": "queued",
            "background": True,
            "output": [],
        }

        if request_body.get("stream"):
            created = {**response, "status": "in_progress"}
            completed = {**response, "status": "completed"}
            events = [
                {
                    "type": "response.created",
                    "sequence_number": 0,
                    "response": created,
                },
                {
                    "type": "response.completed",
                    "sequence_number": 1,
                    "response": completed,
                },
            ]
            payload = b"".join(
                f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode()
                for event in events
            ) + b"data: [DONE]\n\n"
            content_type = "text/event-stream"
        else:
            payload = json.dumps(response).encode()
            content_type = "application/json"

        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)
        self.wfile.flush()

    def _send_prompt_response(self, request_body):
        """Serve an OpenAI-owned prompt-template response."""
        response = {
            "id": f"resp_sdk_prompt_{time.time_ns()}",
            "object": "response",
            "created_at": int(time.time()),
            "model": request_body.get("model", "gpt-5"),
            "status": "completed",
            "background": False,
            "output": [],
        }

        if request_body.get("stream"):
            created = {**response, "status": "in_progress"}
            events = [
                {
                    "type": "response.created",
                    "sequence_number": 0,
                    "response": created,
                },
                {
                    "type": "response.completed",
                    "sequence_number": 1,
                    "response": response,
                },
            ]
            payload = b"".join(
                f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode()
                for event in events
            ) + b"data: [DONE]\n\n"
            content_type = "text/event-stream"
        else:
            payload = json.dumps(response).encode()
            content_type = "application/json"

        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)
        self.wfile.flush()

    def _send_conversation_response(self, request_body):
        """Serve native Responses without a model so append/hydration is deterministic."""
        request_input = json.dumps(request_body.get("input"))
        local_tool_limit = "STREAM-LOCAL-DELETE-410" in request_input
        web_call_limit = "STREAM-WEB-LIMIT-410" in request_input
        tool_model = request_body["model"] == "sdk-conversation-tool-stream"
        if tool_model:
            type(self).tool_round_count += 1
        tool_first_round = tool_model and type(self).tool_round_count == 1
        if "STREAM-CONCURRENT-410" in request_input:
            barrier = type(self).concurrent_barrier
            assert barrier is not None
            barrier.wait(timeout=15)
        response_id = f"resp_sdk_conv_{time.time_ns()}"
        type(self).last_response_id = response_id
        response = {
            "id": response_id,
            "object": "response",
            "created_at": int(time.time()),
            "model": request_body["model"],
            "status": "completed",
            "output": (
                [
                    {
                        "id": f"ws_{response_id}",
                        "type": "web_search_call",
                        "status": "completed",
                        "action": {"type": "search", "queries": ["test"]},
                    }
                ]
                if local_tool_limit or tool_first_round
                else [
                    {
                        "id": f"msg_{response_id}",
                        "type": "message",
                        "status": "completed",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "SDK tool answer" if tool_model else "SDK streamed turn",
                            }
                        ],
                    }
                ]
            ),
        }
        if "BUDGET-CANONICAL-INCOMPLETE-410" in request_input:
            response["status"] = "incomplete"
            response["incomplete_details"] = {"reason": "max_output_tokens"}
            response["output"][0]["status"] = "incomplete"
            response["output"][0]["content"][0]["text"] = "partial answer"
        if web_call_limit:
            # The full-flow example allows 32 web calls per round. A completed
            # upstream snapshot with 33 calls makes IRR emit an SSE error in
            # place of the deferred response.completed terminal.
            response["output"] = [
                {
                    "id": f"ws_{response_id}_{index}",
                    "type": "web_search_call",
                    "status": "completed",
                    "action": {"type": "search", "queries": ["test"]},
                }
                for index in range(33)
            ]
        if request_body.get("stream"):
            created = {**response, "status": "in_progress", "output": []}
            if web_call_limit:
                # Exceed the one-byte replay limit plus decoder headroom before
                # IRR replaces the held completed terminal with an SSE error.
                created["instructions"] = "x" * 3072
            frames = [
                {"type": "response.created", "sequence_number": 0, "response": created},
                {"type": "response.completed", "sequence_number": 1, "response": response},
            ]
            encoded_frames = [
                f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode()
                for event in frames
            ]
            payload = b"".join(encoded_frames) + b"data: [DONE]\n\n"
            content_type = "text/event-stream"
        else:
            payload = json.dumps(response).encode()
            content_type = "application/json"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        chunked = "BUDGET-CHUNKED-410" in request_input and not request_body.get("stream")
        if chunked:
            self.send_header("Transfer-Encoding", "chunked")
        else:
            self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        if request_body.get("stream") and (
            "STREAM-DELETE-410" in request_input
            or "STREAM-DISCONNECT-410" in request_input
            or local_tool_limit
        ):
            self.wfile.write(encoded_frames[0])
            self.wfile.flush()
            gate = type(self).terminal_gate
            assert gate is not None
            assert gate.wait(timeout=15), "test did not release the terminal event"
            self.wfile.write(encoded_frames[1] + b"data: [DONE]\n\n")
        else:
            if chunked:
                for start in range(0, len(payload), 64):
                    part = payload[start : start + 64]
                    self.wfile.write(f"{len(part):x}\r\n".encode() + part + b"\r\n")
                self.wfile.write(b"0\r\n\r\n")
            else:
                self.wfile.write(payload)
        self.wfile.flush()

    def do_POST(self):
        self._forward()

    def do_GET(self):
        self._forward()


class NativeCompactionBackendHandler(BaseHTTPRequestHandler):
    """Deterministic native Responses backend for SDK rehydration coverage."""

    requests: ClassVar[list[dict]] = []
    response_id_prefix: ClassVar[str] = "resp_provider_compaction_sdk_"

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b""
        request_body = json.loads(body)
        type(self).requests.append(request_body)
        request_number = len(type(self).requests)
        output = []
        if request_number == 1:
            output = [
                {
                    "type": "compaction",
                    "id": "cmp_provider_sdk",
                    "encrypted_content": "provider-opaque-state",
                }
            ]
        payload = json.dumps(
            {
                "id": f"{type(self).response_id_prefix}{request_number}",
                "object": "response",
                "created_at": 1,
                "model": VLLM_MODEL,
                "status": "completed",
                "output": output,
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt, *args):
        pass


class SimulatorBackendHandler(BaseHTTPRequestHandler):
    """Record Praxis requests and script hosted-tool Chat responses.

    ``llm-d-inference-sim`` does not deterministically select a tool for
    ``tool_choice=auto`` or stop selecting tools after a result. For the two
    translated hosted tools exercised in simulator mode, this handler acts as
    the backend and returns a deterministic tool call followed by assistant
    text. It never rewrites or re-serializes the request Praxis sent. All other
    requests, including native Responses requests, are forwarded byte-for-byte
    to the simulator.
    """

    recorded_requests: ClassVar[list[tuple[str, dict[str, Any]]]] = []

    def log_message(self, fmt, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b""
        request_body = json.loads(body)
        type(self).recorded_requests.append((self.path, request_body))

        scripted_tool = self._scripted_tool_name(request_body)
        if scripted_tool is not None:
            self._send_scripted_chat_response(request_body, scripted_tool)
            return

        headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() not in ("host", "content-length")
        }
        url = f"{VLLM_BASE_URL.rstrip('/')}{self.path}"
        with httpx.Client(timeout=300.0) as client:
            with client.stream(
                self.command, url, headers=headers, content=body
            ) as upstream:
                self.send_response(upstream.status_code)
                for key, value in upstream.headers.items():
                    if key.lower() in (
                        "transfer-encoding",
                        "content-length",
                        "connection",
                    ):
                        continue
                    self.send_header(key, value)
                self.end_headers()
                for chunk in upstream.iter_raw():
                    if chunk:
                        self.wfile.write(chunk)
                        self.wfile.flush()

    def _scripted_tool_name(self, request_body: dict[str, Any]) -> str | None:
        if not self.path.rstrip("/").endswith("/v1/chat/completions"):
            return None

        tool_names = [
            tool.get("function", {}).get("name")
            for tool in request_body.get("tools", [])
            if isinstance(tool, dict) and tool.get("type") == "function"
        ]
        for hosted_tool in ("web_search", "file_search"):
            if hosted_tool in tool_names:
                return hosted_tool
        return None

    def _send_scripted_chat_response(
        self, request_body: dict[str, Any], tool_name: str
    ) -> None:
        messages = request_body.get("messages", [])
        has_tool_result = any(
            message.get("role") == "tool"
            for message in messages
            if isinstance(message, dict)
        )
        if has_tool_result:
            tool_results = [
                message.get("content", "")
                for message in messages
                if isinstance(message, dict) and message.get("role") == "tool"
            ]
            message = {
                "role": "assistant",
                "content": "Tool result received: " + " ".join(tool_results),
            }
            finish_reason = "stop"
        else:
            query = (
                "latest Praxis Proxy release"
                if tool_name == "web_search"
                else "Praxis marker"
            )
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"call_simulator_{tool_name}",
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps({"query": query}),
                        },
                    }
                ],
            }
            finish_reason = "tool_calls"

        completion = {
            "id": f"chatcmpl_simulator_{time.time_ns()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request_body.get("model", VLLM_MODEL),
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 10,
                "total_tokens": 20,
            },
        }
        payload = json.dumps(completion).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def _assert_simulator_auto_tool_round(
    request_start: int,
    *,
    tool_name: str,
) -> None:
    """Assert the exact Chat requests Praxis emitted for a scripted round."""
    recorded = [
        body
        for path, body in SimulatorBackendHandler.recorded_requests[request_start:]
        if path.rstrip("/").endswith("/v1/chat/completions")
        and any(
            tool.get("function", {}).get("name") == tool_name
            for tool in body.get("tools", [])
            if isinstance(tool, dict)
        )
    ]
    assert len(recorded) == 2, (
        f"expected exactly two {tool_name} Chat requests; got {recorded}"
    )
    first, reentry = recorded

    description, max_length = {
        "web_search": ("Search the web for up-to-date information.", 4_096),
        "file_search": (
            "Search the configured vector stores for relevant files.",
            65_536,
        ),
    }[tool_name]
    expected_tools = [
        {
            "type": "function",
            "function": {
                "name": tool_name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": max_length,
                        }
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        }
    ]

    for round_name, request_body in (("first", first), ("re-entry", reentry)):
        assert request_body.get("tool_choice") == "auto", (
            f"{round_name} request must preserve tool_choice='auto'; "
            f"got {request_body.get('tool_choice')!r} in {request_body}"
        )
        assert request_body.get("tools") == expected_tools, (
            f"{round_name} request has incorrect declared tools: {request_body}"
        )

    first_roles = [message.get("role") for message in first.get("messages", [])]
    reentry_roles = [
        message.get("role") for message in reentry.get("messages", [])
    ]
    assert "tool" not in first_roles, first
    assert "tool" in reentry_roles, reentry



def _write_agentic_config(
    praxis_port: int,
    db_path: str,
    mcp_port: int,
    search_port: int,
    *,
    translate_to_chat: bool = False,
    backend_endpoint: str | None = None,
    real_web_search: bool = False,
) -> str:
    """Patch agentic-loop.yaml for mocked or credentialed agentic tests."""
    config = _load_example_config(AGENTIC_CONFIG_PATH, praxis_port)
    vllm = backend_endpoint if translate_to_chat else _vllm_endpoint()
    if vllm is None:
        raise ValueError("translated agentic config requires a backend endpoint")
    config = config.replace('- "127.0.0.1:3001"', f'- "{vllm}"')
    if config.count("read_timeout_ms:") != 1:
        raise RuntimeError(
            "agentic-loop.yaml must declare exactly one cluster read_timeout_ms; "
            "the vLLM harness no longer injects a second copy"
        )
    # agentic-loop.yaml is the canonical unified config (#1046): it wires all
    # three request-phase dispatchers (web_search, mcp_dispatch,
    # file_search_callout) under the single agentic-loop owner. Retarget the
    # file-search vector store at OGX so the file-search dispatcher is live here
    # too; it stays inert for web/mcp-only tests that emit no file_search_call.
    config = config.replace("http://127.0.0.1:8001", f"http://{_ogx_endpoint()}")
    config = _patch_store_backend(config, db_path)
    # The loopback MCP callout's SSRF posture is governed by
    # ``insecure_options.allow_private_upstreams`` (no per-filter opt-in), which
    # agentic-loop.yaml already enables -- so no injection is needed here.
    config = config.replace(
        "max_iterations: 11\n",
        # agentic-loop.yaml already sets the IRR's overall ``timeout_ms``;
        # inject only ``step_timeout_ms`` here to avoid a duplicate key.
        "max_iterations: 11\n"
        "        step_timeout_ms: 300000\n",
    )
    configured_web_search = (
        "- filter: openai_web_search\n"
        "                provider: brave\n"
        "                api_key: ${WEB_SEARCH_API_KEY}"
    )
    if real_web_search:
        replacement_web_search = (
            "- filter: openai_web_search\n"
            "                provider: tavily\n"
            "                api_key: ${TAVILY_API_KEY}"
        )
    else:
        replacement_web_search = (
            "- filter: openai_web_search\n"
            "                provider: brave\n"
            "                api_key: test-key\n"
            f"                base_url: http://127.0.0.1:{search_port}"
        )
    if configured_web_search not in config:
        raise RuntimeError("agentic-loop.yaml web-search block changed")
    config = config.replace(configured_web_search, replacement_web_search, 1)
    # agentic-loop.yaml already declares ``allow_private_upstreams: true`` in its
    # ``insecure_options``, which is the operator opt-in the executor's SSRF check
    # requires for the loopback provider callout — no test-time injection needed.
    if translate_to_chat:
        config = config.replace(
            "              - filter: openai_responses_proxy\n"
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

    path = _persist_config(config)
    return path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def backend_endpoint():
    """Return the live backend or the recording simulator backend."""
    if VLLM_TEST_BACKEND == "live":
        yield _vllm_endpoint()
        return

    SimulatorBackendHandler.recorded_requests = []
    port = _free_port()
    server = HTTPServer(("127.0.0.1", port), SimulatorBackendHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"127.0.0.1:{port}"
    finally:
        server.shutdown()
        thread.join()


@pytest.fixture(scope="session")
def praxis_proxy(tmp_path_factory, request):
    """Start a Praxis proxy backed by vLLM for the test session."""
    port = _free_port()
    db_dir = tmp_path_factory.mktemp("responses")
    db_path = str(db_dir / "responses.db")
    config_path = _write_full_flow_config(port, db_path)
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
def compression_proxy(tmp_path_factory, request):
    """Start a Praxis proxy whose response store has zstd compression enabled."""
    port = _free_port()
    db_dir = tmp_path_factory.mktemp("responses-compression")
    db_path = str(db_dir / "responses.db")
    config_path = _write_full_flow_config(port, db_path, compression=True)
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
                    f"\n=== Compression store Praxis logs ===\n{f.read()}",
                    file=sys.stderr,
                )
        os.unlink(config_path)


@pytest.fixture(scope="session")
def irr_streaming_proxy(tmp_path_factory, request):
    """Start a Praxis proxy with terminal Responses streaming through IRR."""
    port = _free_port()
    config_path = _write_example_config(
        IRR_STREAMING_CONFIG_PATH, port, backend_endpoint=_vllm_endpoint()
    )
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
def chat_streaming_proxy(tmp_path_factory, request, backend_endpoint):
    """Start the Responses-to-Chat streaming example."""
    port = _free_port()
    db_dir = tmp_path_factory.mktemp("responses-chat-streaming")
    db_path = str(db_dir / "responses.db")
    config_path = _write_example_config(
        CHAT_STREAMING_CONFIG_PATH,
        port,
        backend_endpoint=backend_endpoint,
        db_path=db_path,
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
                    f"\n=== Chat streaming Praxis logs ===\n{f.read()}",
                    file=sys.stderr,
                )
        os.unlink(config_path)


@pytest.fixture(scope="session")
def reasoning_proxy(tmp_path_factory, request):
    """Start the reasoning-dialect example against live vLLM."""
    port = _free_port()
    db_dir = tmp_path_factory.mktemp("responses-reasoning")
    db_path = str(db_dir / "responses.db")
    config_path = _write_example_config(
        REASONING_CONFIG_PATH,
        port,
        backend_endpoint=_vllm_endpoint(),
        db_path=db_path,
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
                    f"\n=== Reasoning Praxis logs ===\n{f.read()}",
                    file=sys.stderr,
                )
        os.unlink(config_path)


@pytest.fixture(scope="session")
def client_tool_compat_proxy(tmp_path_factory, request):
    """Start the client-tool-compat example against live vLLM."""
    port = _free_port()
    db_dir = tmp_path_factory.mktemp("client-tool-compat")
    db_path = str(db_dir / "responses.db")
    config_path = _write_example_config(
        CLIENT_TOOL_COMPAT_CONFIG_PATH,
        port,
        backend_endpoint=_vllm_endpoint(),
        db_path=db_path,
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
                    f"\n=== Client tool compat Praxis logs ===\n{f.read()}",
                    file=sys.stderr,
                )
        os.unlink(config_path)


@pytest.fixture(scope="session")
def client_tool_compat_client(client_tool_compat_proxy):
    """Return an SDK client using the client-tool-compat pipeline."""
    return _make_openai_client(client_tool_compat_proxy)


@pytest.fixture(scope="session")
def client_tool_compat_chat_proxy(tmp_path_factory, request):
    """Start the composed client-tool-compat + Chat Completions example (#1206)."""
    port = _free_port()
    db_dir = tmp_path_factory.mktemp("client-tool-compat-chat")
    db_path = str(db_dir / "responses.db")
    config_path = _write_example_config(
        CLIENT_TOOL_COMPAT_CHAT_CONFIG_PATH,
        port,
        backend_endpoint=_vllm_endpoint(),
        db_path=db_path,
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
                    f"\n=== Client tool compat (Chat) Praxis logs ===\n{f.read()}",
                    file=sys.stderr,
                )
        os.unlink(config_path)


@pytest.fixture(scope="session")
def client_tool_compat_chat_client(client_tool_compat_chat_proxy):
    """Return an SDK client using the composed compat + Chat Completions pipeline."""
    return _make_openai_client(client_tool_compat_chat_proxy)


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
    """Start the compact example against inference and the mock summarizer."""
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


def _witness_proxy_session(
    tmp_path_factory, request, search_port=None, max_event_bytes=None, retained_limit=None,
    skip_agentic_response=False,
):
    """Start a proxy whose native backend is a recording shim in front of vLLM.

    Shared generator body for the witness fixtures. Yields ``(client,
    forwarded_bodies)`` where ``forwarded_bodies`` accumulates the JSON bodies
    the vLLM Responses backend receives, letting a test assert on the request
    the proxy actually forwards upstream after its rewrites.
    """
    ResponsesWitnessHandler.forwarded_bodies = []
    ResponsesWitnessHandler.tool_round_count = 0
    ResponsesWitnessHandler.last_response_id = None
    forwarded = ResponsesWitnessHandler.forwarded_bodies
    backend_port = _free_port()
    server = ThreadingHTTPServer(("127.0.0.1", backend_port), ResponsesWitnessHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    port = _free_port()
    db_dir = tmp_path_factory.mktemp("responses-witness")
    db_path = str(db_dir / "responses.db")
    config_path = _write_full_flow_config(
        port,
        db_path,
        backend_endpoint=f"127.0.0.1:{backend_port}",
        search_port=search_port,
        max_event_bytes=max_event_bytes,
        retained_limit=retained_limit,
        skip_agentic_response=skip_agentic_response,
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
        client = _make_openai_client(port, default_headers=TRUSTED_OWNER_HEADERS)
        yield client, forwarded
    finally:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        log_file.close()
        server.shutdown()
        ResponsesWitnessHandler.concurrent_barrier = None
        ResponsesWitnessHandler.terminal_gate = None
        if not started or request.session.testsfailed > 0:
            with open(log_path) as f:
                print(
                    f"\n=== Witness backend Praxis logs ===\n{f.read()}",
                    file=sys.stderr,
                )
        os.unlink(config_path)


@pytest.fixture()
def witness_backend_client(tmp_path_factory, request):
    """Function-scoped witness proxy with the stock rehydrate config."""
    yield from _witness_proxy_session(tmp_path_factory, request)


@pytest.fixture()
def witness_budgeted_conversation_client(tmp_path_factory, request):
    """Full-flow SQLite witness with room for the response but not append staging."""
    yield from _witness_proxy_session(tmp_path_factory, request, retained_limit=8_192)


@pytest.fixture()
def witness_noncanonical_budgeted_conversation_client(tmp_path_factory, request):
    """A selected inference response skips agentic canonicalization under a budget."""
    yield from _witness_proxy_session(
        tmp_path_factory,
        request,
        retained_limit=67_108_864,
        skip_agentic_response=True,
    )


@pytest.fixture()
def witness_budgeted_continuation_client(tmp_path_factory, request):
    """Full-flow SQLite witness at the default 64 MiB retained allowance."""
    yield from _witness_proxy_session(tmp_path_factory, request, retained_limit=67_108_864)


@pytest.fixture()
def witness_tool_client(tmp_path_factory, request, search_server):
    """Full-flow witness with a deterministic hosted web-search endpoint."""
    yield from _witness_proxy_session(tmp_path_factory, request, search_server)


@pytest.fixture()
def witness_replay_limited_tool_client(tmp_path_factory, request, search_server):
    """Hosted web-search witness whose Response replay limit is one byte."""
    yield from _witness_proxy_session(
        tmp_path_factory, request, search_server, max_event_bytes=1
    )


class DirectBudgetBackendHandler(BaseHTTPRequestHandler):
    """Serve fixed native Responses bodies without running the IRR step."""

    protocol_version = "HTTP/1.1"
    terminal_gate: ClassVar[threading.Event | None] = None

    def log_message(self, fmt, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        request_body = json.loads(self.rfile.read(length))
        model = request_body["model"]
        if model in {"terminal-stream", "terminal-stream-no-event", "sequence-stream"}:
            response = {
                "id": "resp_direct_budget_terminal_stream",
                "object": "response",
                "created_at": 1780000000,
                "model": model,
                "status": "completed",
                "output": [],
            }
            if model == "sequence-stream":
                terminal = (
                    "event: response.in_progress\ndata: "
                    + json.dumps({
                        "type": "response.in_progress", "sequence_number": 7,
                    }, separators=(",", ":"))
                    + "\n\n"
                ).encode()
            else:
                event_prefix = "event: response.completed\n" if model == "terminal-stream" else ""
                terminal = (
                    event_prefix + "data: "
                    + json.dumps({
                        "type": "response.completed",
                        "sequence_number": 1,
                        "response": response,
                    }, separators=(",", ":"))
                    + "\n\n"
                ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(terminal)
            self.wfile.flush()
            gate = type(self).terminal_gate
            if gate is not None:
                gate.wait(timeout=10)
            # This unfinished SSE comment exceeds the direct restore's 64 KiB
            # retained allowance after the terminal frame is already on wire.
            try:
                self.wfile.write(b":" + b"x" * 120_000)
                self.wfile.flush()
                time.sleep(0.2)
                self.wfile.write(b"\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            self.close_connection = True
            return
        size = 50 if model == "chunked" else int(model)
        response = {
            "id": f"resp_direct_budget_{model}",
            "object": "response",
            "created_at": 1780000000,
            "model": model,
            "status": "completed",
            "output": [{
                "id": "msg_1", "type": "message", "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "x" * size}],
            }],
        }
        payload = json.dumps(response, separators=(",", ":")).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        if model == "chunked":
            self.send_header("Transfer-Encoding", "chunked")
        else:
            self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        if model == "chunked":
            self.wfile.write(f"{len(payload):x}\r\n".encode() + payload + b"\r\n0\r\n\r\n")
        else:
            self.wfile.write(payload)
        self.wfile.flush()
        self.close_connection = True


@pytest.fixture()
def direct_budget_client(tmp_path, request):
    """Route a Responses request around IRR while retaining its loop policy."""
    backend = ThreadingHTTPServer(("127.0.0.1", 0), DirectBudgetBackendHandler)
    thread = threading.Thread(target=backend.serve_forever, daemon=True)
    thread.start()
    proxy_port = _free_port()
    config = _load_example_config(
        "examples/configs/openai/responses/agentic-loop-fixture.yaml",
        proxy_port,
        backend_endpoint=f"127.0.0.1:{backend.server_port}",
    )
    budget_param = getattr(request, "param", 16_384)
    budget_limit, store_response_first = (
        budget_param if isinstance(budget_param, tuple) else (budget_param, False)
    )
    config = config.replace(
        "max_retained_bytes: 67108864", f"max_retained_bytes: {budget_limit}"
    )
    if store_response_first:
        # Put Store on the first response callback while Rehydrate still
        # establishes the request's retained state before the direct branch.
        store_start = config.index("      - filter: openai_response_store\n")
        rehydrate = "      - filter: openai_responses_rehydrate\n"
        rehydrate_start = config.index(rehydrate, store_start)
        config = (
            config[:store_start]
            + rehydrate
            + config[store_start:rehydrate_start]
            + config[rehydrate_start + len(rehydrate):]
        )
    anchor = "      - filter: iterative_request_router\n"
    assert config.count(anchor) == 1
    config = config.replace(
        anchor,
        "      - filter: request_id\n"
        "        branch_chains:\n"
        "          - name: direct\n"
        "            rejoin: terminal\n"
        "            chains:\n"
        "              - name: direct-inline\n"
        "                filters:\n"
        "                  - filter: router\n"
        "                    routes:\n"
        "                      - path_prefix: \"/\"\n"
        "                        cluster: direct-backend\n"
        "                  - filter: load_balancer\n"
        "                    clusters:\n"
        "                      - name: direct-backend\n"
        f"                        endpoints: [\"127.0.0.1:{backend.server_port}\"]\n"
        + anchor,
    )
    config = _patch_store_backend(config, str(tmp_path / "responses.db"))
    admin_port = _free_port()
    while admin_port == proxy_port:
        admin_port = _free_port()
    config += f'\nadmin:\n  address: "127.0.0.1:{admin_port}"\n'
    config_path = _persist_config(config)
    log_path = str(tmp_path / "praxis.log")
    with open(log_path, "w") as log_file:
        proc = subprocess.Popen(
            [_find_binary(), "-c", config_path],
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        try:
            _wait_for_proxy(proxy_port, proc, log_path)
            yield _make_openai_client(proxy_port, default_headers=TRUSTED_OWNER_HEADERS)
        finally:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            if request.session.testsfailed > 0:
                print(_read_log_tail(log_path), file=sys.stderr)
            backend.shutdown()
            os.unlink(config_path)


def test_direct_budget_store_rejects_before_success_headers(direct_budget_client):
    """A noncanonical direct branch reports a structured budget failure."""
    client = direct_budget_client
    with pytest.raises(APIStatusError) as known:
        client.responses.create(model="3000", input="hello", store=True)
    assert known.value.status_code == 502
    assert known.value.response.json()["error"]["type"] == "server_error"

    with pytest.raises(APIStatusError) as unknown:
        client.responses.create(model="chunked", input="hello", store=True)
    assert unknown.value.status_code == 502
    assert unknown.value.response.json()["error"]["type"] == "server_error"

    passthrough = client.responses.create(model="3000", input="hello", store=False)
    assert passthrough.status == "completed"
    assert passthrough.output[0].content[0].text == "x" * 3000


@pytest.mark.parametrize("direct_budget_client", [32_768], indirect=True)
def test_direct_budget_store_admits_bounded_known_length(direct_budget_client):
    """A known safe direct body still reaches persistence with budget headroom."""
    admitted = direct_budget_client.responses.create(model="10", input="hello", store=True)
    assert admitted.status == "completed"
    assert admitted.output[0].content[0].text == "x" * 10


@pytest.mark.parametrize("direct_budget_client", [65_536], indirect=True)
def test_direct_budget_store_persists_bounded_previous_response(direct_budget_client):
    """A stored direct continuation survives both restoration and persistence."""
    client = direct_budget_client
    seed = client.responses.create(model="10", input="seed", store=True)
    continued = client.responses.create(
        model="20", input="next", previous_response_id=seed.id, store=True
    )
    assert continued.status == "completed"
    assert continued.previous_response_id == seed.id
    retrieved = client.responses.retrieve(continued.id)
    assert retrieved.previous_response_id == seed.id


@pytest.mark.parametrize("direct_budget_client", [32_768], indirect=True)
@pytest.mark.parametrize("model", ["8000", "chunked"])
@pytest.mark.parametrize("store", [False, True])
def test_direct_budget_restore_rejects_before_success_headers(direct_budget_client, model, store):
    """A direct previous-response rewrite reports budget failures as HTTP 502."""
    client = direct_budget_client
    seed = client.responses.create(model="10", input="seed", store=True)
    continued = client.responses.create(
        model="20", input="next", previous_response_id=seed.id, store=False
    )
    assert continued.status == "completed"
    assert continued.previous_response_id == seed.id

    with pytest.raises(APIStatusError) as failed:
        client.responses.create(
            model=model, input="next", previous_response_id=seed.id, store=store
        )
    assert failed.value.status_code == 502
    assert failed.value.response.json()["error"]["type"] == "server_error"


@pytest.mark.parametrize("direct_budget_client", [65_536], indirect=True)
@pytest.mark.parametrize("model", ["terminal-stream", "terminal-stream-no-event"])
def test_direct_budget_stream_overflow_after_completion_has_one_terminal(direct_budget_client, model):
    """The direct SSE route closes after a late overflow without an error event."""
    seed = direct_budget_client.responses.create(model="10", input="seed", store=True)
    url = f"{str(direct_budget_client.base_url).rstrip('/')}/responses"
    wire = bytearray()
    transport_failed = False
    terminal_marker = b"event: response.completed" if model == "terminal-stream" else b'"type":"response.completed"'
    gate = threading.Event()
    DirectBudgetBackendHandler.terminal_gate = gate
    try:
        with httpx.Client(timeout=15.0) as client:
            try:
                with client.stream(
                    "POST", url,
                    headers={"Content-Type": "application/json", **TRUSTED_OWNER_HEADERS},
                    json={
                        "model": model, "input": "next", "stream": True,
                        "store": False, "previous_response_id": seed.id,
                    },
                ) as response:
                    assert response.status_code == 200
                    for chunk in response.iter_raw():
                        wire.extend(chunk)
                        if terminal_marker in wire and b"\n\n" in wire:
                            gate.set()  # The provider may now send the oversized tail.
            except (httpx.RemoteProtocolError, httpx.ReadError):
                transport_failed = True  # The filter aborts the committed transport.
    finally:
        gate.set()
        DirectBudgetBackendHandler.terminal_gate = None
    assert transport_failed, "the over-budget stream must end with a transport error"
    assert wire.count(terminal_marker) == 1
    assert b"event: error" not in wire
    assert b"event: response.failed" not in wire
    assert b"x" * 128 not in wire, "the rejected comment cannot leak downstream"


@pytest.mark.parametrize("direct_budget_client", [(65_536, True)], indirect=True)
def test_direct_store_stream_overflow_after_completion_closes_wire(direct_budget_client):
    """Store cannot append an error after forwarding a direct SSE completion."""
    seed = direct_budget_client.responses.create(model="10", input="seed", store=True)
    url = f"{str(direct_budget_client.base_url).rstrip('/')}/responses"
    wire = bytearray()
    transport_failed = False
    gate = threading.Event()
    DirectBudgetBackendHandler.terminal_gate = gate
    try:
        with httpx.Client(timeout=15.0) as client:
            try:
                with client.stream(
                    "POST", url,
                    headers={"Content-Type": "application/json", **TRUSTED_OWNER_HEADERS},
                    json={
                        "model": "terminal-stream", "input": "next", "stream": True,
                        "store": True, "previous_response_id": seed.id,
                    },
                ) as response:
                    assert response.status_code == 200
                    for chunk in response.iter_raw():
                        wire.extend(chunk)
                        if b"event: response.completed" in wire and b"\n\n" in wire:
                            gate.set()
            except (httpx.RemoteProtocolError, httpx.ReadError):
                transport_failed = True
    finally:
        gate.set()
        DirectBudgetBackendHandler.terminal_gate = None
    assert transport_failed, "late Store exhaustion must abort the committed stream"
    assert wire.count(b"event: response.completed") == 1
    assert b"event: error" not in wire
    assert b"event: response.failed" not in wire
    assert b"x" * 128 not in wire
    with pytest.raises(NotFoundError):
        direct_budget_client.responses.retrieve("resp_direct_budget_terminal_stream")


@pytest.mark.parametrize("direct_budget_client", [(65_536, True)], indirect=True)
def test_direct_store_stream_budget_error_follows_provider_sequence(direct_budget_client):
    """A local Store error follows the last admitted provider SSE sequence."""
    seed = direct_budget_client.responses.create(model="10", input="seed", store=True)
    url = f"{str(direct_budget_client.base_url).rstrip('/')}/responses"
    wire = bytearray()
    gate = threading.Event()
    DirectBudgetBackendHandler.terminal_gate = gate
    try:
        with httpx.Client(timeout=15.0) as client:
            with client.stream(
                "POST", url,
                headers={"Content-Type": "application/json", **TRUSTED_OWNER_HEADERS},
                json={
                    "model": "sequence-stream", "input": "next", "stream": True,
                    "store": True, "previous_response_id": seed.id,
                },
            ) as response:
                assert response.status_code == 200
                for chunk in response.iter_raw():
                    wire.extend(chunk)
                    if b"event: response.in_progress" in wire and b"\n\n" in wire:
                        gate.set()
    finally:
        gate.set()
        DirectBudgetBackendHandler.terminal_gate = None
    assert wire.count(b"event: response.in_progress") == 1
    assert wire.count(b"event: error") == 1
    assert b'"sequence_number":8' in wire
    assert b"x" * 128 not in wire


@pytest.fixture()
def provider_compaction_client(tmp_path_factory, request):
    """Function-scoped native Responses backend with a provider compaction."""
    NativeCompactionBackendHandler.requests = []
    NativeCompactionBackendHandler.response_id_prefix = "resp_provider_compaction_sdk_"
    requests = NativeCompactionBackendHandler.requests
    backend_port = _free_port()
    server = HTTPServer(("127.0.0.1", backend_port), NativeCompactionBackendHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    port = _free_port()
    db_dir = tmp_path_factory.mktemp("responses-provider-compaction")
    db_path = str(db_dir / "responses.db")
    config_path = _write_full_flow_config(
        port, db_path, backend_endpoint=f"127.0.0.1:{backend_port}"
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
        client = _make_openai_client(port, default_headers=TRUSTED_OWNER_HEADERS)
        yield client, requests
    finally:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        log_file.close()
        server.shutdown()
        if not started or request.session.testsfailed > 0:
            with open(log_path) as f:
                print(
                    f"\n=== Provider compaction Praxis logs ===\n{f.read()}",
                    file=sys.stderr,
                )
        os.unlink(config_path)


@pytest.fixture()
def usage_less_compact_client(tmp_path_factory, request, compaction_server):
    """Native Responses continuation with no stored usage and an armed budget."""
    NativeCompactionBackendHandler.requests = []
    # PostgreSQL SDK runs share one database across these two fixtures.
    NativeCompactionBackendHandler.response_id_prefix = "resp_usage_less_compaction_sdk_"
    CompactionHandler.requests = []
    backend_port = _free_port()
    backend = HTTPServer(("127.0.0.1", backend_port), NativeCompactionBackendHandler)
    backend_thread = threading.Thread(target=backend.serve_forever, daemon=True)
    backend_thread.start()

    port = _free_port()
    db_dir = tmp_path_factory.mktemp("responses-usage-less-compact")
    db_path = str(db_dir / "responses.db")
    config_path = _write_full_flow_config(
        port,
        db_path,
        backend_endpoint=f"127.0.0.1:{backend_port}",
        compact_callout_port=compaction_server,
        retained_limit=131_072,
    )
    log_path = str(db_dir / "praxis.log")
    log_file = open(log_path, "w")
    proc = subprocess.Popen(
        [_find_binary(), "-c", config_path],
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    try:
        _wait_for_proxy(port, proc, log_path)
        yield _make_openai_client(port, default_headers=TRUSTED_OWNER_HEADERS), db_path
    finally:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        log_file.close()
        backend.shutdown()
        if request.session.testsfailed > 0:
            with open(log_path) as f:
                print(f"\n=== Usage-less compact Praxis logs ===\n{f.read()}", file=sys.stderr)
        os.unlink(config_path)


def _reasoning_capture_session(tmp_path_factory, request):
    """Start the reasoning example with a capturing mock Chat backend.

    Yields ``(client, captured_bodies)`` where ``captured_bodies`` accumulates
    the Chat Completions request bodies the backend receives. A mock backend
    (rather than live vLLM) keeps the assertion deterministic and independent of
    model output: the test checks the assistant reasoning field forwarded upstream.
    """
    ChatCaptureHandler.captured_bodies = []
    captured = ChatCaptureHandler.captured_bodies
    backend_port = _free_port()
    server = HTTPServer(("127.0.0.1", backend_port), ChatCaptureHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    port = _free_port()
    db_dir = tmp_path_factory.mktemp("responses-reasoning-capture")
    db_path = str(db_dir / "responses.db")
    config_path = _write_reasoning_backend_config(
        port, db_path, backend_port, getattr(request, "param", "vllm"),
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
        client = _make_openai_client(port)
        yield client, captured
    finally:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        log_file.close()
        server.shutdown()
        if not started or request.session.testsfailed > 0:
            with open(log_path) as f:
                print(
                    f"\n=== Reasoning capture Praxis logs ===\n{f.read()}",
                    file=sys.stderr,
                )
        os.unlink(config_path)


@pytest.fixture()
def reasoning_capture_client(tmp_path_factory, request):
    """Function-scoped reasoning proxy with a capturing mock Chat backend."""
    yield from _reasoning_capture_session(tmp_path_factory, request)


@pytest.fixture(scope="session")
def openai_client(praxis_proxy):
    """Return an OpenAI client pointed at the local Praxis proxy."""
    return _make_openai_client(
        praxis_proxy,
        default_headers={
            **TRUSTED_OWNER_HEADERS,
            "x-user-ogx-key": "Bearer test",
        },
    )


@pytest.fixture(scope="session")
def other_owner_openai_client(praxis_proxy):
    """Return a same-tenant Responses client with another subject."""
    return _make_openai_client(
        praxis_proxy,
        default_headers={**TRUSTED_OWNER_HEADERS, "x-auth-user": "other-test-user"},
    )


@pytest.fixture(scope="session")
def compression_openai_client(compression_proxy):
    """Return an OpenAI client pointed at the compression-enabled proxy."""
    return _make_openai_client(
        compression_proxy, default_headers=TRUSTED_OWNER_HEADERS
    )


@pytest.fixture(scope="session")
def irr_streaming_client(irr_streaming_proxy):
    """Return an OpenAI client using the terminal-streaming IRR proxy."""
    return _make_openai_client(irr_streaming_proxy)


@pytest.fixture(scope="session")
def chat_streaming_client(chat_streaming_proxy):
    """Return an SDK client using Responses-to-Chat stream translation."""
    return _make_openai_client(chat_streaming_proxy)


@pytest.fixture(scope="session")
def reasoning_client(reasoning_proxy):
    """Return an SDK client using the reasoning-dialect example."""
    return _make_openai_client(reasoning_proxy)


@pytest.fixture(scope="session")
def compact_client(compact_proxy):
    """Return an SDK client using the compact filter example."""
    return _make_openai_client(compact_proxy)


@pytest.fixture(scope="session")
def web_search_chat_streaming_proxy(
    tmp_path_factory, request, search_server, backend_endpoint
):
    """Start the streaming web-search-through-Chat example."""
    port = _free_port()
    db_dir = tmp_path_factory.mktemp("web-search-chat-streaming")
    config_path = _write_web_search_chat_streaming_config(
        port, search_server, backend_endpoint
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
        yield port, search_server
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
                    f"\n=== Web search chat streaming Praxis logs ===\n{f.read()}",
                    file=sys.stderr,
                )
        os.unlink(config_path)


@pytest.fixture(scope="session")
def web_search_chat_streaming_client(web_search_chat_streaming_proxy):
    """Return an SDK client using streaming web-search-through-Chat translation."""
    proxy_port, _ = web_search_chat_streaming_proxy
    return _make_openai_client(proxy_port)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestOpenAIResponsesStoreCompression:
    """Integration tests for response store with zstd payload compression enabled."""

    def test_compressed_store_and_retrieve(self, compression_openai_client):
        client = compression_openai_client
        response = client.responses.create(
            model=VLLM_MODEL,
            input="Say exactly: COMPRESSED-OK. /no_think",
            temperature=0,
            store=True,
            max_output_tokens=512,
        )

        assert response.status == "completed"
        assert response.id

        retrieved = client.responses.retrieve(response.id)

        assert retrieved.id == response.id
        assert retrieved.status == "completed"
        # The full response object survives the compress -> BLOB -> decompress
        # trip unchanged.
        assert retrieved.output_text == response.output_text
        _assert_usage(retrieved.usage)

    def test_compressed_input_items_round_trip(self, compression_openai_client):
        client = compression_openai_client
        response = client.responses.create(
            model=VLLM_MODEL,
            input=[
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "The marker is COMPRESSED-INPUT-OK.",
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

        # Reading the stored input items back decompresses the `input` column;
        # the original marker text must be intact.
        items = client.responses.input_items.list(response.id, order="asc")
        texts = [
            block.text
            for item in items.data
            if item.type == "message"
            for block in item.content
            if block.type == "input_text"
        ]
        assert any("COMPRESSED-INPUT-OK" in text for text in texts), texts

    def test_compressed_previous_response_chaining(
        self, compression_openai_client
    ):
        client = compression_openai_client
        first = client.responses.create(
            model=VLLM_MODEL,
            input="Say exactly: FIRST-TURN-OK. /no_think",
            temperature=0,
            store=True,
            max_output_tokens=512,
        )
        assert first.status == "completed"

        # Chaining rehydrates first's compressed messages/input on the read
        # path before the next turn is assembled.
        second = client.responses.create(
            model=VLLM_MODEL,
            input="Say exactly: SECOND-TURN-OK. /no_think",
            previous_response_id=first.id,
            temperature=0,
            store=True,
            max_output_tokens=512,
        )
        assert second.status == "completed"
        assert second.id != first.id


class TestOpenAIResponsesVLLM:
    """Responses API integration tests against the selected backend."""

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
        assert exc_info.value.status_code == 404, "store=false response must not be retrievable"

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

    def test_same_tenant_other_owner_cannot_access_response(
        self, openai_client, other_owner_openai_client
    ):
        response = openai_client.responses.create(
            model=VLLM_MODEL,
            input="Say exactly: OWNER-PRIVATE /no_think",
            temperature=0,
            store=True,
            max_output_tokens=128,
        )

        with pytest.raises(NotFoundError):
            other_owner_openai_client.responses.retrieve(response.id)
        with pytest.raises(NotFoundError):
            other_owner_openai_client.responses.input_items.list(response.id)
        with pytest.raises(NotFoundError):
            other_owner_openai_client.responses.delete(response.id)
        with pytest.raises(BadRequestError):
            other_owner_openai_client.responses.create(
                model=VLLM_MODEL,
                input="This must not use another owner's state.",
                previous_response_id=response.id,
                store=True,
            )

        assert openai_client.responses.retrieve(response.id).id == response.id

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
        assert exc_info.value.status_code == 404, "retrieving a missing response must return 404"

        with pytest.raises(NotFoundError) as exc_info:
            openai_client.responses.input_items.list(missing_id)
        assert exc_info.value.status_code == 404, "listing input items for a missing response must return 404"

        with pytest.raises(NotFoundError) as exc_info:
            openai_client.responses.delete(missing_id)
        assert exc_info.value.status_code == 404, "deleting a missing response must return 404"

    def test_invalid_previous_response_id_is_rejected(self, openai_client):
        with pytest.raises(BadRequestError) as exc_info:
            openai_client.responses.create(
                model=VLLM_MODEL,
                input="This request must not reach vLLM.",
                previous_response_id="resp_missing_sdk_integration",
                store=True,
            )
        assert exc_info.value.status_code == 400, "unknown previous_response_id must return 400"
        assert "resp_missing_sdk_integration" in str(exc_info.value), "error must name the missing previous_response_id"

    def test_streaming_validation_failure_returns_json_not_sse(self, openai_client):
        """Issue #1001: a request that fails pre-stream validation must return
        the JSON ``{"error": {...}}`` envelope with a non-2xx status -- never a
        nonconforming ``text/event-stream`` SSE error event on an uncommitted
        stream -- even when the caller set ``stream: true``.

        OpenAI raises the typed error from the JSON body before opening the
        stream, so the official client surfaces this as a ``BadRequestError``
        rather than a live event stream. Praxis matches that transport: locally
        generated pre-commitment rejections always use ``application/json``.
        This is the streaming sibling of
        ``test_invalid_previous_response_id_is_rejected`` and the live-backend
        counterpart of the Rust unit coverage in the ``responses::error`` and
        ``responses::validate`` modules.
        """
        # The official SDK surfaces the pre-stream failure as a typed error, not
        # a stream object -- proving it parsed a JSON error body, not an SSE one.
        with pytest.raises(BadRequestError) as exc_info:
            openai_client.responses.create(
                model=VLLM_MODEL,
                input="This request must not reach vLLM.",
                previous_response_id="resp_missing_sdk_integration",
                stream=True,
                store=True,
            )
        assert exc_info.value.status_code == 400, "stream:true rejection must return 400"
        assert "resp_missing_sdk_integration" in str(exc_info.value), "error must name the missing previous_response_id"

        # Assert the wire shape precisely: a stream:true rejection must be an
        # application/json error envelope, not an SSE error event.
        raw = httpx.post(
            f"{str(openai_client.base_url).rstrip('/')}/responses",
            headers={"Authorization": "Bearer test", **TRUSTED_OWNER_HEADERS},
            json={
                "model": VLLM_MODEL,
                "input": "This request must not reach vLLM.",
                "previous_response_id": "resp_missing_sdk_integration",
                "stream": True,
                "store": True,
            },
            timeout=10,
        )
        assert raw.status_code == 400, "stream:true pre-stream rejection must return 400"
        content_type = raw.headers.get("content-type", "")
        assert content_type.startswith("application/json"), (
            "a stream:true pre-stream rejection must use application/json, not "
            f"text/event-stream; got: {content_type!r}"
        )
        error = raw.json()["error"]
        assert isinstance(error["message"], str) and error["message"], "error envelope must carry a non-empty message string"
        assert isinstance(error["type"], str) and error["type"], "error envelope must carry a non-empty type string"
        assert "resp_missing_sdk_integration" in error["message"], "error message must name the missing previous_response_id"

    def test_malformed_request_has_sdk_compatible_error(self, openai_client):
        response = httpx.post(
            f"{str(openai_client.base_url).rstrip('/')}/responses",
            headers={"Authorization": "Bearer test", **TRUSTED_OWNER_HEADERS},
            json={},
            timeout=10,
        )
        assert response.status_code == 400, "malformed request must return 400"
        error = response.json()["error"]
        assert isinstance(error["message"], str), "error envelope message must be a string"
        assert error["message"], "error envelope message must be non-empty"
        assert isinstance(error["type"], str), "error envelope type must be a string"
        assert error["type"], "error envelope type must be non-empty"

    def test_invalid_input_container_is_rejected(self, openai_client):
        with pytest.raises(BadRequestError) as exc_info:
            openai_client.responses.create(
                model=VLLM_MODEL,
                input=["not-an-input-item"],
                store=False,
            )
        assert exc_info.value.status_code == 400, "invalid input item must return 400"

    @pytest.mark.critical_vllm
    @requires_real_inference
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
        inference backend never sees it and echoes previous_response_id: null.
        The rehydrate filter restores the caller's id into the response body so
        the client always sees the id it sent, per the Responses API contract.

        This assertion is metadata-only and independent of model output, so it
        is deterministic with either the simulator or a real backend.

        Manifest linkage: this is the SDK regression counterpart of the
        committed synthetic inference fixture -- coverage feature
        ``responses.native.continuation``, scenario
        ``responses/native-continuation`` (see
        tests/integration/fixtures/inference/). No live recording is committed
        for that feature -- it stays ``synthetic_only`` because a live recording
        requires explicit authorization -- so this SDK test provides the
        gateway-level confidence instead.
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

    def test_buffered_store_false_conversation_appends_before_body(self, witness_backend_client):
        """The full-flow IRR has canonical output when the outer header hook runs."""
        client, _ = witness_backend_client
        conversation = client.conversations.create()
        try:
            response = client.responses.create(
                model="sdk-conversation-stream",
                input="BUFFERED-STORE-FALSE-410",
                conversation=conversation.id,
                store=False,
            )
            assert response.status == "completed"
            items = client.conversations.items.list(conversation.id, order="asc")
            assert [item.role for item in items.data if item.type == "message"] == [
                "user",
                "assistant",
            ]
            assert "BUFFERED-STORE-FALSE-410" in items.data[0].content[0].text
        finally:
            client.conversations.delete(conversation.id)

    def test_budgeted_noncanonical_conversation_rejects_before_success_headers(
        self, witness_noncanonical_budgeted_conversation_client
    ):
        """A skipped agentic response has unknown append cost at the 200 header.

        The safe policy rejects even a small body when the aggregate budget is
        armed, because SQL/cache staging cannot be proven before commitment.
        """
        client, forwarded = witness_noncanonical_budgeted_conversation_client
        conversation = client.conversations.create()
        try:
            with pytest.raises(APIStatusError) as exc_info:
                client.responses.create(
                    model="sdk-conversation-stream",
                    input="NONCANONICAL-APPEND-HEADER-410",
                    conversation=conversation.id,
                    store=False,
                )
            assert exc_info.value.status_code == 502
            assert exc_info.value.response.json()["error"]["type"] == "server_error"
            assert forwarded, "the backend response must reach the header hook"
            assert client.conversations.items.list(conversation.id).data == []
        finally:
            client.conversations.delete(conversation.id)

    @pytest.mark.parametrize("store", [False, True])
    def test_budgeted_canonical_incomplete_conversation_keeps_partial_response(
        self, witness_budgeted_continuation_client, store
    ):
        """A finalized incomplete response needs no append and keeps its 200 wire."""
        client, _ = witness_budgeted_continuation_client
        conversation = client.conversations.create()
        try:
            response = client.responses.create(
                model="sdk-conversation-stream",
                input="BUDGET-CANONICAL-INCOMPLETE-410",
                conversation=conversation.id,
                store=store,
                max_output_tokens=1,
            )
            assert response.status == "incomplete"
            assert response.output_text == "partial answer"
            assert client.conversations.items.list(conversation.id).data == []
        finally:
            client.conversations.delete(conversation.id)

    def test_budgeted_previous_response_id_continuation_keeps_small_completion(
        self, witness_budgeted_continuation_client
    ):
        """A rehydrate buffer ceiling must not reject an admitted small body."""
        client, _ = witness_budgeted_continuation_client
        first = client.responses.create(
            model="sdk-conversation-stream", input="BUDGET-PREVIOUS-BASE-410", store=True
        )
        assert first.status == "completed"
        second = client.responses.create(
            model="sdk-conversation-stream",
            input="BUDGET-PREVIOUS-NEXT-410",
            previous_response_id=first.id,
            store=True,
        )
        assert second.status == "completed"
        assert second.previous_response_id == first.id

    def test_buffered_store_true_conversation_persists_before_append(self, witness_backend_client):
        """Header persistence and append both complete before buffered delivery."""
        client, _ = witness_backend_client
        conversation = client.conversations.create()
        try:
            response = client.responses.create(
                model="sdk-conversation-stream",
                input="BUFFERED-STORE-TRUE-410",
                conversation=conversation.id,
                store=True,
            )
            assert response.status == "completed"
            assert client.responses.retrieve(response.id).id == response.id
            items = client.conversations.items.list(conversation.id, order="asc")
            assert [item.role for item in items.data if item.type == "message"] == [
                "user",
                "assistant",
            ]
        finally:
            client.conversations.delete(conversation.id)

    def test_budgeted_chunked_store_true_conversation_rejects_before_commit(
        self, witness_budgeted_conversation_client
    ):
        """A chunked native completion returns Responses 502 without durable success."""
        client, _ = witness_budgeted_conversation_client
        conversation = client.conversations.create()
        try:
            with pytest.raises(APIStatusError) as exc_info:
                client.responses.create(
                    model="sdk-conversation-stream",
                    input="BUDGET-CHUNKED-410",
                    conversation=conversation.id,
                    store=True,
                )
            assert exc_info.value.status_code == 502
            assert exc_info.value.response.json()["error"]["type"] == "server_error"
            assert client.conversations.items.list(conversation.id).data == []
            response_id = ResponsesWitnessHandler.last_response_id
            assert response_id is not None
            with pytest.raises(APIStatusError) as retrieval:
                client.responses.retrieve(response_id)
            assert retrieval.value.status_code == 404
        finally:
            client.conversations.delete(conversation.id)

    def test_streamed_conversation_append_and_follow_up(self, witness_backend_client):
        """The shipped full-flow graph appends a streamed turn before completion.

        The witness backend supplies deterministic native SSE. The same test
        runs with SQLite locally and PostgreSQL when DATABASE_URL is set.
        """
        client, forwarded = witness_backend_client
        conversation = client.conversations.create(metadata={"retained": "yes"})
        try:
            foreign_client = OpenAI(
                base_url=client.base_url,
                api_key="test",
                default_headers={**TRUSTED_OWNER_HEADERS, "x-auth-user": "other-user"},
                max_retries=0,
            )
            try:
                with pytest.raises(BadRequestError):
                    foreign_client.responses.create(
                        model="sdk-conversation-stream",
                        input="foreign turn",
                        conversation=conversation.id,
                        stream=True,
                    )
            finally:
                foreign_client.close()

            events = list(
                client.responses.create(
                    model="sdk-conversation-stream",
                    input="STREAM-FIRST-410",
                    conversation=conversation.id,
                    stream=True,
                    store=True,
                )
            )
            completed = [event for event in events if event.type == "response.completed"]
            assert len(completed) == 1
            first = completed[0].response
            assert first.status == "completed"
            assert [item.type for item in first.output] == ["message"]
            assert first.output[0].id == f"msg_{first.id}"
            assert first.output[0].content[0].text == "SDK streamed turn"
            assert client.responses.retrieve(first.id).output == first.output

            items = client.conversations.items.list(conversation.id, order="asc")
            assert [item.role for item in items.data if item.type == "message"] == [
                "user",
                "assistant",
            ]
            assert len(items.data) == 2
            assert "STREAM-FIRST-410" in json.dumps(
                [item.model_dump() for item in items.data], default=str
            )
            retrieved_conversation = client.conversations.retrieve(conversation.id)
            assert retrieved_conversation.metadata == {"retained": "yes"}
            assert retrieved_conversation.created_at == conversation.created_at

            before_follow_up = len(forwarded)
            second = client.responses.create(
                model="sdk-conversation-stream",
                input="STREAM-SECOND-410",
                conversation=conversation.id,
                store=True,
            )
            assert second.status == "completed"
            assert len(forwarded) == before_follow_up + 1
            replay = json.dumps(forwarded[-1]["input"])
            assert "STREAM-FIRST-410" in replay
            assert "SDK streamed turn" in replay
            assert "STREAM-SECOND-410" in replay

            items = client.conversations.items.list(conversation.id, order="asc")
            assert [item.role for item in items.data if item.type == "message"] == [
                "user",
                "assistant",
                "user",
                "assistant",
            ]

            unstored = list(
                client.responses.create(
                    model="sdk-conversation-stream",
                    input="STREAM-UNSTORED-410",
                    conversation=conversation.id,
                    stream=True,
                    store=False,
                )
            )
            unstored_completed = [event for event in unstored if event.type == "response.completed"]
            assert len(unstored_completed) == 1
            with pytest.raises(NotFoundError):
                client.responses.retrieve(unstored_completed[0].response.id)
            items = client.conversations.items.list(conversation.id, order="asc")
            assert [item.role for item in items.data if item.type == "message"] == [
                "user", "assistant", "user", "assistant", "user", "assistant"
            ]

            # Hold both backend completions until both requests have read the
            # same Conversation snapshot. Their inserts must then allocate
            # distinct positions atomically instead of overwriting one turn.
            ResponsesWitnessHandler.concurrent_barrier = threading.Barrier(2)

            def concurrent_turn(marker):
                worker = OpenAI(
                    base_url=client.base_url,
                    api_key="test",
                    default_headers=TRUSTED_OWNER_HEADERS,
                    max_retries=0,
                    timeout=30,
                )
                try:
                    events = list(
                        worker.responses.create(
                            model="sdk-conversation-stream",
                            input=f"STREAM-CONCURRENT-410-{marker}",
                            conversation=conversation.id,
                            stream=True,
                            store=True,
                        )
                    )
                    completed = [
                        event for event in events if event.type == "response.completed"
                    ]
                    assert len(completed) == 1
                    return completed[0].response.id
                finally:
                    worker.close()

            try:
                with ThreadPoolExecutor(max_workers=2) as executor:
                    ids = list(executor.map(concurrent_turn, ("A", "B")))
            finally:
                ResponsesWitnessHandler.concurrent_barrier = None
            assert len(set(ids)) == 2
            items = client.conversations.items.list(
                conversation.id, order="asc", limit=20
            )
            assert len(items.data) == 10
            payload = json.dumps([item.model_dump() for item in items.data], default=str)
            assert "STREAM-CONCURRENT-410-A" in payload
            assert "STREAM-CONCURRENT-410-B" in payload
        finally:
            client.conversations.delete(conversation.id)

    def test_streamed_hosted_tool_appends_once_and_rehydrates(self, witness_tool_client):
        """A real IRR search round appends one canonical tool item and replays its turn."""
        client, forwarded = witness_tool_client
        conversation = client.conversations.create()
        searches_before = BraveSearchHandler.request_count
        try:
            events = list(
                client.responses.create(
                    model="sdk-conversation-tool-stream",
                    input="STREAM-TOOL-410",
                    conversation=conversation.id,
                    stream=True,
                    store=True,
                    tools=[{"type": "web_search"}],
                    max_tool_calls=1,
                    extra_headers={"x-user-brave-key": "controlled-test-only"},
                )
            )
            completed = [event for event in events if event.type == "response.completed"]
            assert len(completed) == 1
            output = completed[0].response.output
            assert [item.type for item in output] == ["web_search_call", "message"]
            assert BraveSearchHandler.request_count == searches_before + 1
            assert len(forwarded) == 2, "one tool dispatch should produce one re-entry"
            assert "Mock Search Result" in json.dumps(forwarded[1]["input"])
            bridge = [
                item
                for item in forwarded[1]["input"]
                if isinstance(item, dict)
                and item.get("type") in {"function_call", "function_call_output"}
            ]
            assert [item["type"] for item in bridge] == [
                "function_call",
                "function_call_output",
            ], "the backend history keeps the web-search bridge in call/result order"
            assert bridge[0]["call_id"] == bridge[1]["call_id"]

            items = client.conversations.items.list(conversation.id, order="asc")
            assert [item.type for item in items.data] == [
                "message",
                "web_search_call",
                "message",
            ]
            assert items.data[0].role == "user"
            assert items.data[2].role == "assistant"
            assert items.data[1].id == output[0].id

            follow_up = client.responses.create(
                model="sdk-conversation-stream",
                input="STREAM-TOOL-FOLLOW-UP-410",
                conversation=conversation.id,
                store=True,
            )
            assert follow_up.status == "completed"
            replay = json.dumps(forwarded[-1]["input"])
            assert "STREAM-TOOL-410" in replay
            assert "SDK tool answer" in replay
            items = client.conversations.items.list(conversation.id, order="asc")
            assert [item.type for item in items.data] == [
                "message",
                "web_search_call",
                "message",
                "message",
                "message",
            ]
        finally:
            client.conversations.delete(conversation.id)

    @qualification_profile("supporting")
    @pytest.mark.parametrize("replay_limited", [False, True], ids=["default", "replay-limit"])
    def test_failed_irr_stream_does_not_append_completed_upstream_snapshot(
        self, request, replay_limited
    ):
        """An IRR error must not hydrate a model snapshot hidden from the client."""
        fixture = (
            "witness_replay_limited_tool_client" if replay_limited else "witness_tool_client"
        )
        client, forwarded = request.getfixturevalue(fixture)
        conversation = client.conversations.create()
        searches_before = BraveSearchHandler.request_count
        try:
            events = list(
                client.responses.create(
                    model="sdk-conversation-stream",
                    input="STREAM-WEB-LIMIT-410",
                    conversation=conversation.id,
                    stream=True,
                    store=True,
                    tools=[{"type": "web_search"}],
                    extra_headers={"x-user-brave-key": "controlled-test-only"},
                )
            )
            assert any(event.type == "error" for event in events)
            assert not any(event.type == "response.completed" for event in events)
            assert BraveSearchHandler.request_count == searches_before
            assert len(forwarded) == 1
            assert client.conversations.items.list(conversation.id).data == []
            with pytest.raises(NotFoundError):
                client.responses.retrieve(events[0].response.id)

            follow_up = client.responses.create(
                model="sdk-conversation-stream",
                input="STREAM-WEB-LIMIT-FOLLOW-UP-410",
                conversation=conversation.id,
                store=True,
            )
            assert follow_up.status == "completed"
            replay = json.dumps(forwarded[-1]["input"])
            assert "STREAM-WEB-LIMIT-410" not in replay
            assert "STREAM-WEB-LIMIT-FOLLOW-UP-410" in replay
            assert "web_search_call" not in replay
        finally:
            client.conversations.delete(conversation.id)

    @pytest.mark.parametrize("client_close_pause", [0.75, 1.5])
    def test_streamed_client_close_can_commit_completed_turn(
        self, witness_backend_client, client_close_pause
    ):
        """An unobserved close may commit a terminal the SDK never consumed."""
        client, forwarded = witness_backend_client
        conversation = client.conversations.create()
        gate = threading.Event()
        ResponsesWitnessHandler.terminal_gate = gate
        try:
            stream = client.responses.create(
                model="sdk-conversation-stream",
                input="STREAM-DISCONNECT-410",
                conversation=conversation.id,
                stream=True,
                store=True,
            )
            first = next(iter(stream))
            assert first.type == "response.created"
            stream.close()
            time.sleep(client_close_pause)
            gate.set()

            deadline = time.monotonic() + 5
            while True:
                items = client.conversations.items.list(conversation.id, order="asc")
                if len(items.data) == 2 or time.monotonic() >= deadline:
                    break
                time.sleep(0.1)
            assert [item.role for item in items.data] == ["user", "assistant"]
            assert client.responses.retrieve(first.response.id).status == "completed"
            follow_up = client.responses.create(
                model="sdk-conversation-stream",
                input="STREAM-DISCONNECT-FOLLOW-UP-410",
                conversation=conversation.id,
                store=True,
            )
            assert follow_up.status == "completed"
            replay = json.dumps(forwarded[-1]["input"])
            assert "STREAM-DISCONNECT-410" in replay
            assert "SDK streamed turn" in replay
        finally:
            gate.set()
            ResponsesWitnessHandler.terminal_gate = None
            client.conversations.delete(conversation.id)

    def test_streamed_append_failure_withholds_committed_terminal(
        self, witness_backend_client
    ):
        """An append failure after early SSE delivery must abort before completion."""
        client, _ = witness_backend_client
        conversation = client.conversations.create()
        gate = threading.Event()
        ResponsesWitnessHandler.terminal_gate = gate
        stream = None
        try:
            stream = client.responses.create(
                model="sdk-conversation-stream",
                input="STREAM-DELETE-410",
                conversation=conversation.id,
                stream=True,
                store=True,
            )
            first = next(iter(stream))
            assert first.type == "response.created"
            client.conversations.delete(conversation.id)
            gate.set()

            observed = []
            try:
                for event in stream:
                    observed.append(event)
            except (APIConnectionError, httpx.RemoteProtocolError, httpx.ReadError):
                pass
            assert not any(event.type == "response.completed" for event in observed)
            assert client.responses.retrieve(first.response.id).status == "completed"
        finally:
            gate.set()
            ResponsesWitnessHandler.terminal_gate = None
            if stream is not None:
                stream.close()

    def test_streamed_local_completion_append_failure_withholds_terminal(
        self, witness_backend_client
    ):
        """A request-side tool-limit completion must append before its SSE terminal."""
        client, _ = witness_backend_client
        conversation = client.conversations.create()
        gate = threading.Event()
        ResponsesWitnessHandler.terminal_gate = gate
        stream = None
        try:
            stream = client.responses.create(
                model="sdk-conversation-stream",
                input="STREAM-LOCAL-DELETE-410",
                conversation=conversation.id,
                stream=True,
                store=True,
                tools=[{"type": "web_search"}],
                max_tool_calls=0,
                extra_headers={"x-user-brave-key": "controlled-test-only"},
            )
            first = next(iter(stream))
            assert first.type == "response.created"
            client.conversations.delete(conversation.id)
            gate.set()

            observed = []
            try:
                for event in stream:
                    observed.append(event)
            except (APIConnectionError, httpx.RemoteProtocolError, httpx.ReadError):
                pass
            assert not any(event.type == "response.completed" for event in observed)
            assert client.responses.retrieve(first.response.id).status == "completed"
        finally:
            gate.set()
            ResponsesWitnessHandler.terminal_gate = None
            if stream is not None:
                stream.close()

    def test_sdk_replays_provider_compaction_on_previous_response_id(
        self, provider_compaction_client
    ):
        """Native provider compaction survives SDK-driven local rehydration."""
        client, forwarded = provider_compaction_client
        first = client.responses.create(
            model=VLLM_MODEL,
            input="Start the provider-compaction conversation.",
            store=True,
        )
        second = client.responses.create(
            model=VLLM_MODEL,
            input="Continue after provider compaction.",
            previous_response_id=first.id,
            store=True,
        )

        assert second.status == "completed"
        assert len(forwarded) == 2, forwarded
        replayed = forwarded[1]
        assert replayed.get("previous_response_id") is None, replayed
        assert any(
            item.get("type") == "compaction"
            and item.get("id") == "cmp_provider_sdk"
            and item.get("encrypted_content") == "provider-opaque-state"
            for item in replayed.get("input", [])
        ), replayed

    def test_input_tokens_history_read_with_agentic_policy(
        self, provider_compaction_client
    ):
        """A count operation reads stored history without create-state."""
        client, forwarded = provider_compaction_client
        first = client.responses.create(
            model=VLLM_MODEL,
            input="Store history for the token-count operation.",
            store=True,
        )
        assert first.status == "completed"

        response = httpx.post(
            f"{str(client.base_url).rstrip('/')}/responses/input_tokens",
            headers={"Authorization": "Bearer test", **TRUSTED_OWNER_HEADERS},
            json={
                "model": VLLM_MODEL,
                "input": "Count this continuation.",
                "previous_response_id": first.id,
            },
            timeout=10,
        )
        # This shipped example has no input_tokens backend and deliberately
        # returns its static unsupported-operation response after rehydration.
        # A 413 here means the bounded history read failed before that route.
        assert response.status_code == 404, response.text
        assert response.json()["error"]["message"] == "unsupported managed Responses operation"
        assert len(forwarded) == 1, forwarded

    def test_truncation_forwarded_to_backend_through_rehydration(
        self, witness_backend_client
    ):
        """Issue #532: the caller's ``truncation`` must reach the native backend
        on both a fresh turn and a rehydrated continuation.

        The unit test only proves ``truncation`` survives into ``ResponsesState``
        before the proxy rewrites the outbound body. This drives the full native
        pipeline against a recording shim in front of vLLM to prove the backend
        actually *receives* the value on both turns:

        * turn 1 sends ``truncation="auto"`` — forwarded verbatim (no rewrite);
        * turn 2 rehydrates via ``previous_response_id`` and sends
          ``truncation="disabled"`` — the proxy replays history into ``input``
          and strips ``previous_response_id``, but must still forward the
          caller's ``truncation``.

        Guarding both spec values through the rewrite path is exactly what #532
        requires: process conversation history without dropping client-supplied
        request settings.
        """
        client, forwarded = witness_backend_client

        before_first = len(forwarded)
        first = client.responses.create(
            model=VLLM_MODEL,
            input="Remember this nonce: TRUNCATE-4821. Acknowledge it. /no_think",
            temperature=0,
            truncation="auto",
            store=True,
            max_output_tokens=128,
        )
        assert first.status == "completed"

        first_seen = forwarded[before_first:]
        assert first_seen, "backend received no request on the first turn"
        first_backend = first_seen[-1]
        assert first_backend.get("truncation") == "auto", first_backend

        before_second = len(forwarded)
        second = client.responses.create(
            model=VLLM_MODEL,
            input="What nonce did I tell you? Repeat it exactly. /no_think",
            temperature=0,
            previous_response_id=first.id,
            truncation="disabled",
            store=True,
            max_output_tokens=128,
        )
        assert second.status == "completed"

        second_seen = forwarded[before_second:]
        assert second_seen, "backend received no request on the rehydrated turn"
        second_backend = second_seen[-1]
        # The caller's truncation survives the outbound-body rewrite...
        assert second_backend.get("truncation") == "disabled", second_backend
        # ...while previous_response_id is stripped and history is replayed
        # into `input` instead.
        assert second_backend.get("previous_response_id") is None, second_backend
        assert isinstance(second_backend.get("input"), list), second_backend

    @requires_real_inference
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

    @requires_real_inference
    def test_conversation_multi_turn_append_back(self, openai_client):
        conversation = openai_client.conversations.create()
        try:
            first = openai_client.responses.create(
                model=VLLM_MODEL,
                input="Remember the color ultramarine. /no_think",
                conversation={"id": conversation.id},
                store=True,
                temperature=0,
                # Qwen3 is a hybrid thinking model and its /no_think soft switch
                # is not honored through this backend, so it emits a reasoning
                # block before answering. Budget enough output tokens for the
                # reasoning plus the short answer so the turn completes instead
                # of truncating to status "incomplete".
                max_output_tokens=2048,
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
                max_output_tokens=2048,
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

    def test_streaming_conversation_append_back_with_store_false(self, openai_client):
        """The SDK observes a completed SSE turn and its local Conversation items."""
        conversation = openai_client.conversations.create()
        try:
            events = _collect_stream(
                openai_client.responses.create(
                    model=VLLM_MODEL,
                    input=[{"role": "user", "content": "Reply with a short greeting. /no_think"}],
                    conversation=conversation.id,
                    stream=True,
                    store=False,
                    temperature=0,
                    max_output_tokens=2048,
                )
            )
            completed = [
                event.response
                for event in events
                if event.type == "response.completed"
            ]
            assert len(completed) == 1, [event.type for event in events]
            assert completed[0].status == "completed"
            assert completed[0].output_text

            items = openai_client.conversations.items.list(conversation.id, order="asc")
            messages = [item for item in items.data if item.type == "message"]
            assert [item.role for item in messages] == ["user", "assistant"]
            assert "short greeting" in messages[0].content[0].text
            assert messages[1].content[0].text == completed[0].output_text, (
                "the appended assistant item must match the streamed terminal output"
            )
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
        assert exc_info.value.status_code == 400, "unknown conversation must return 400"
        assert "conv_00000000000000000000000000000000" in str(exc_info.value), "error must name the missing conversation id"

    def test_streaming_rehydrated_response_echoes_previous_response_id(
        self, openai_client
    ):
        """Issue #932 (streaming half): a rehydrated ``stream=True`` turn must
        echo the caller's previous_response_id back inside the SSE lifecycle
        frames.

        Streaming sibling of
        ``test_rehydrated_response_echoes_previous_response_id``. Same contract,
        same full-flow pipeline (store -> stream_events -> rehydrate), but
        stream=True: the proxy replays prior turns via the ``input`` array and
        strips previous_response_id from the upstream request, so the backend
        streams ``previous_response_id: null`` in every response-lifecycle
        frame. The rehydrate filter restores the caller's id into each lifecycle
        frame as it streams -- without buffering the stream -- so the client's
        terminal ``response.completed`` event carries the id it sent.

        The assertions are metadata-only and independent of model output, so
        they stay deterministic with either the simulator or a real backend.

        Manifest linkage: this is the SDK regression counterpart of the
        committed synthetic inference fixture -- coverage feature
        ``responses.native.continuation``, scenario
        ``responses/native-continuation-stream`` (see
        tests/integration/fixtures/inference/). No live recording is committed
        for that feature -- it stays ``synthetic_only`` because a live recording
        requires explicit authorization -- so this SDK test provides the
        gateway-level confidence for the streaming path.
        """
        first = openai_client.responses.create(
            model=VLLM_MODEL,
            input="Say exactly: STREAM-ECHO-BASE /no_think",
            store=True,
            max_output_tokens=128,
        )

        assert first.status == "completed"
        assert first.id

        stream = openai_client.responses.create(
            model=VLLM_MODEL,
            input="Say exactly: STREAM-ECHO-NEXT /no_think",
            previous_response_id=first.id,
            store=True,
            stream=True,
            max_output_tokens=128,
        )

        event_types = []
        lifecycle_previous_ids = []
        final_response = None

        for event in stream:
            event_types.append(event.type)
            # Only response-lifecycle events (created, in_progress, completed)
            # carry a full response resource; delta/item events do not.
            response_obj = getattr(event, "response", None)
            if response_obj is not None:
                lifecycle_previous_ids.append(
                    getattr(response_obj, "previous_response_id", None)
                )
            if event.type == "response.completed":
                final_response = event.response

        assert event_types[0] == "response.created", event_types
        assert event_types[-1] == "response.completed", event_types
        assert final_response is not None, (
            "stream must terminate with a response.completed event; "
            f"got: {event_types}"
        )
        assert final_response.status == "completed", final_response.status

        # Core #932 streaming contract: the terminal event the client observes
        # carries the caller's previous_response_id, not the backend's null.
        assert final_response.previous_response_id == first.id, (
            "the streamed terminal response must echo the caller's "
            "previous_response_id even though the proxy strips it from the "
            "rehydrated upstream request; got: "
            f"{final_response.previous_response_id!r}"
        )

        # The restore rewrites every response-lifecycle frame incrementally, so
        # no streamed frame may still echo the backend's null id.
        assert lifecycle_previous_ids, (
            "the stream must contain at least one response-lifecycle frame; "
            f"got event types: {event_types}"
        )
        assert all(pid == first.id for pid in lifecycle_previous_ids), (
            "every streamed lifecycle frame must carry the restored "
            f"previous_response_id; got: {lifecycle_previous_ids}"
        )

        # Issue #1150: the persisted record (served by GET) must agree with the
        # terminal frame the client observed. The streaming persistence source is
        # an independent ResponsesState.response_object that the incremental wire
        # rewrite never touches, so before the fix the stored response echoed the
        # backend's null even though the streamed terminal carried first.id.
        retrieved = _retrieve_with_retry(openai_client, final_response.id)
        assert retrieved.previous_response_id == first.id, (
            "the stored streaming response must persist the caller's "
            "previous_response_id, matching the terminal frame the client saw; "
            f"got: {retrieved.previous_response_id!r}"
        )

    def test_streaming_replay_returns_stored_events_in_order(self, openai_client):
        """Local Responses SSE replay: ``GET /v1/responses/{id}?stream=true`` on a
        completed response created with ``stream=true`` replays the exact stored
        event log -- the same normalized events, in original sequence order,
        terminating in the same terminal event -- without reconstructing deltas.

        Metadata-only and provider-neutral: asserts the ``(type,
        sequence_number)`` sequence and terminal identity, not model text, so it
        stays deterministic on the simulator or a real backend.
        """
        live_events = _collect_stream(
            openai_client.responses.create(
                model=VLLM_MODEL,
                input="Say exactly: REPLAY-OK /no_think",
                store=True,
                stream=True,
                max_output_tokens=128,
            )
        )
        terminal = _assert_stream_contract(live_events, require_usage=False)
        response_id = terminal.id
        live_seq = [(event.type, event.sequence_number) for event in live_events]

        # The record and its event log persist as the proxy finishes the body.
        _retrieve_with_retry(openai_client, response_id)

        replay_events = _collect_replay_with_retry(openai_client, response_id)
        replay_seq = [(event.type, event.sequence_number) for event in replay_events]

        # Replay serves the identical normalized events the client observed live,
        # in the same order -- not a reconstruction.
        assert replay_seq == live_seq, (replay_seq, live_seq)
        assert replay_events[0].type == "response.created", replay_seq
        assert replay_events[-1].type in TERMINAL_RESPONSE_EVENTS, replay_seq
        assert replay_events[-1].response.id == response_id
        numbers = [event.sequence_number for event in replay_events]
        assert numbers == sorted(numbers), numbers

    def test_streaming_replay_starting_after_skips_earlier_events(
        self, openai_client
    ):
        """``starting_after=N`` replays only events whose ``sequence_number > N``,
        while still delivering the terminal event."""
        live_events = _collect_stream(
            openai_client.responses.create(
                model=VLLM_MODEL,
                input="Say exactly: REPLAY-CURSOR /no_think",
                store=True,
                stream=True,
                max_output_tokens=128,
            )
        )
        response_id = _assert_stream_contract(live_events, require_usage=False).id
        _retrieve_with_retry(openai_client, response_id)

        full = _collect_replay_with_retry(openai_client, response_id)
        assert len(full) >= 2, [event.type for event in full]

        cursor = full[0].sequence_number
        after = _collect_replay_with_retry(
            openai_client, response_id, starting_after=cursor
        )

        assert after, "replay after the first event must still return events"
        assert all(event.sequence_number > cursor for event in after), [
            event.sequence_number for event in after
        ]
        assert [event.sequence_number for event in after] == [
            event.sequence_number for event in full if event.sequence_number > cursor
        ]
        # The terminal event always survives the cursor so a resumed reader still
        # observes completion.
        assert after[-1].type in TERMINAL_RESPONSE_EVENTS, [
            event.type for event in after
        ]

    def test_streaming_replay_requires_a_stored_event_log(self, openai_client):
        """A response created without ``stream=true`` has no replay log; replaying
        it returns a 400 ``invalid_request_error`` -- never a 404 and never a
        fabricated stream reconstructed from the stored JSON."""
        buffered = openai_client.responses.create(
            model=VLLM_MODEL,
            input="Say exactly: NO-REPLAY /no_think",
            store=True,
            stream=False,
            max_output_tokens=128,
        )
        _retrieve_with_retry(openai_client, buffered.id)

        with pytest.raises(BadRequestError) as exc_info:
            openai_client.responses.retrieve(buffered.id, stream=True)
        assert exc_info.value.status_code == 400, "replaying a non-streamed response must return 400"
        assert "replayable" in str(exc_info.value).lower(), str(exc_info.value)

    def test_streaming_replay_starting_after_requires_stream(self, openai_client):
        """``starting_after`` without ``stream=true`` is rejected with a 400 rather
        than silently returning the plain JSON record."""
        live_events = _collect_stream(
            openai_client.responses.create(
                model=VLLM_MODEL,
                input="Say exactly: CURSOR-NEEDS-STREAM /no_think",
                store=True,
                stream=True,
                max_output_tokens=128,
            )
        )
        response_id = _assert_stream_contract(live_events, require_usage=False).id
        _retrieve_with_retry(openai_client, response_id)

        with pytest.raises(BadRequestError) as exc_info:
            openai_client.responses.retrieve(response_id, starting_after=0)
        assert exc_info.value.status_code == 400, "starting_after without stream must return 400"

    def test_streaming_replay_is_owner_scoped(
        self, openai_client, other_owner_openai_client
    ):
        """Replay honors ownership: another owner in the same tenant gets a 404,
        never another owner's stored events, while the owner still replays."""
        live_events = _collect_stream(
            openai_client.responses.create(
                model=VLLM_MODEL,
                input="Say exactly: REPLAY-PRIVATE /no_think",
                store=True,
                stream=True,
                max_output_tokens=128,
            )
        )
        response_id = _assert_stream_contract(live_events, require_usage=False).id
        _retrieve_with_retry(openai_client, response_id)

        with pytest.raises(NotFoundError):
            other_owner_openai_client.responses.retrieve(response_id, stream=True)

        owner_events = _collect_replay_with_retry(openai_client, response_id)
        assert owner_events[-1].type in TERMINAL_RESPONSE_EVENTS

    def test_streaming_replay_removed_on_delete(self, openai_client):
        """Deleting a response removes its replay log: a subsequent replay returns
        404, matching the plain JSON record's lifecycle."""
        live_events = _collect_stream(
            openai_client.responses.create(
                model=VLLM_MODEL,
                input="Say exactly: REPLAY-DELETE /no_think",
                store=True,
                stream=True,
                max_output_tokens=128,
            )
        )
        response_id = _assert_stream_contract(live_events, require_usage=False).id
        _retrieve_with_retry(openai_client, response_id)

        assert _collect_replay_with_retry(openai_client, response_id)
        assert openai_client.responses.delete(response_id) is None

        with pytest.raises(NotFoundError):
            openai_client.responses.retrieve(response_id, stream=True)

    @pytest.mark.parametrize("stream", [False, True], ids=["buffered", "streaming"])
    def test_conflicting_history_selectors_error_shape(self, openai_client, stream):
        with pytest.raises(BadRequestError) as exc_info:
            openai_client.responses.create(
                model=VLLM_MODEL,
                input="next",
                previous_response_id="resp_previous",
                conversation="conv_existing",
                stream=stream,
            )

        error = exc_info.value
        assert error.status_code == 400, "conflicting history selectors must return 400"
        assert error.body == {
            "code": "mutually_exclusive_parameters",
            "message": (
                "Mutually exclusive parameters. Ensure you are only providing "
                "one of: 'previous_response_id' or 'conversation'."
            ),
            "param": None,
            "type": "invalid_request_error",
        }, "error envelope must report mutually_exclusive_parameters"

    def test_background_mode_is_rejected_before_inference(self, openai_client):
        with pytest.raises(BadRequestError) as exc_info:
            openai_client.responses.create(
                model=VLLM_MODEL,
                input="Run this later.",
                background=True,
            )

        error = exc_info.value
        assert error.status_code == 400, "background mode must return 400"
        assert error.body == {
            "code": "invalid_request_error",
            "message": "background mode is not supported",
            "param": None,
            "type": "invalid_request_error",
        }, "error envelope must report background mode is unsupported"

    def test_openai_bound_background_create_and_stream(self, witness_backend_client):
        """The stock full-flow pipeline preserves OpenAI-owned background creates."""
        client, forwarded = witness_backend_client

        queued = client.responses.create(
            model="gpt-5",
            input="SDK finite background passthrough",
            background=True,
            store=True,
        )
        assert queued.status == "queued"
        assert queued.background is True

        stream = client.responses.create(
            model="gpt-5",
            input="SDK streaming background passthrough",
            background=True,
            store=True,
            stream=True,
        )
        events = list(stream)
        assert [event.type for event in events] == [
            "response.created",
            "response.completed",
        ]
        streamed = events[-1].response
        assert streamed.status == "completed"
        assert streamed.background is True

        background_creates = [
            body
            for body in forwarded
            if body.get("background") is True and body.get("model") == "gpt-5"
        ]
        assert len(background_creates) == 2
        assert [body.get("input") for body in background_creates] == [
            "SDK finite background passthrough",
            "SDK streaming background passthrough",
        ]
        assert all(body.get("store") is True for body in background_creates)
        assert {body.get("stream", False) for body in background_creates} == {
            False,
            True,
        }

    def test_prompt_template_policy_follows_provider_binding(
        self, witness_backend_client
    ):
        """Managed routes reject prompt templates; OpenAI-owned routes preserve them."""
        client, forwarded = witness_backend_client
        prompt = {
            "id": "pmpt_sdk_provider_policy",
            "version": "2",
            "variables": {"name": "Ada"},
        }

        with pytest.raises(BadRequestError) as exc_info:
            client.responses.create(
                model=VLLM_MODEL,
                input="This managed request must not reach inference.",
                prompt=prompt,
                store=False,
            )

        error = exc_info.value
        assert error.status_code == 400
        assert error.body == {
            "code": "invalid_request_error",
            "message": (
                "prompt templates are supported only for OpenAI-owned upstreams"
            ),
            "param": None,
            "type": "invalid_request_error",
        }

        response = client.responses.create(
            model="gpt-5",
            input="Use the OpenAI-owned prompt template.",
            prompt=prompt,
            store=False,
        )
        assert response.status == "completed"

        stream = client.responses.create(
            model="gpt-5",
            input="Stream the OpenAI-owned prompt template.",
            prompt=prompt,
            store=False,
            stream=True,
        )
        events = list(stream)
        assert [event.type for event in events] == [
            "response.created",
            "response.completed",
        ]
        assert events[-1].response.status == "completed"

        prompt_creates = [
            body for body in forwarded if body.get("prompt") is not None
        ]
        assert len(prompt_creates) == 2
        assert {body.get("stream", False) for body in prompt_creates} == {
            False,
            True,
        }
        assert all(body["model"] == "gpt-5" for body in prompt_creates)
        assert all(body["prompt"] == prompt for body in prompt_creates)

    @pytest.mark.critical_vllm
    @requires_real_inference
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

    @pytest.mark.critical_vllm
    @requires_real_inference
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

        The unified agentic pipeline's openai_agentic_loop only auto-executes
        hosted tools (file_search, web_search, MCP); a bare client-side
        function_call has no hosted dispatcher, so the loop terminates
        (action=done) and passes the function_call through to the client.
        Validates that vLLM produces a well-formed function_call through the
        proxy.
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

    @pytest.mark.critical_vllm
    @requires_real_inference
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

    @requires_vllm_compat
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

    @requires_vllm_compat
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


class TestResponsesReasoningVLLM:
    """Reasoning-dialect translation exercised through the OpenAI SDK."""

    def test_reasoning_summary_request_is_rejected(self, reasoning_client):
        """vLLM has no safe-summary contract, so a summary request is a 400."""
        with pytest.raises(BadRequestError) as exc_info:
            reasoning_client.responses.create(
                model=VLLM_MODEL,
                input="What is 2+2?",
                reasoning={"summary": "auto"},
                store=False,
                max_output_tokens=64,
            )
        assert exc_info.value.status_code == 400, "reasoning summary request must return 400"

    def test_non_object_reasoning_is_rejected(self, reasoning_client):
        """A proxy-owned `reasoning` field that is neither object nor null is a 400."""
        response = httpx.post(
            f"{str(reasoning_client.base_url).rstrip('/')}/responses",
            headers={"Authorization": "Bearer test"},
            json={
                "model": VLLM_MODEL,
                "input": "What is 2+2?",
                "reasoning": True,
                "store": False,
            },
            timeout=30,
        )
        assert response.status_code == 400, "non-object reasoning must return 400"
        error = response.json()["error"]
        assert error["type"] == "invalid_request_error", "error envelope type must be invalid_request_error"

    def test_reasoning_dialect_promotes_raw_reasoning_to_an_item(
        self, reasoning_client,
    ):
        """A thinking response yields a reasoning item, never a leaked summary."""
        response = reasoning_client.responses.create(
            model=VLLM_MODEL,
            input="What is 2+2? Think briefly, then answer.",
            reasoning={"effort": "low"},
            temperature=0,
            store=True,
            # The continuation contract accepts only completed stored
            # responses. Leave enough room for Qwen3's reasoning block and
            # terminal answer instead of treating a token-capped response as a
            # valid continuation parent.
            max_output_tokens=1024,
        )

        assert response.status == "completed", response.status
        output_types = [item.type for item in response.output]
        assert output_types, "response must carry at least one output item"

        for item in response.output:
            if item.type != "reasoning":
                continue
            # Raw chain-of-thought lives only in the reasoning item content and
            # must never leak into the summary array.
            assert item.summary == [], item.summary
            assert item.content, "reasoning item must carry content"
            assert item.content[0].type == "reasoning_text"
            assert item.content[0].text

        continuation = reasoning_client.responses.create(
            model=VLLM_MODEL, previous_response_id=response.id,
            input="Now give the answer briefly.",
            temperature=0, store=False, max_output_tokens=128,
        )
        assert continuation.status in ("completed", "incomplete"), continuation.status
        assert continuation.output, "stored reasoning continuation must produce output"

    def test_replayed_reasoning_item_uses_the_assistant_reasoning_field(
        self, reasoning_capture_client,
    ):
        """A rehydrated reasoning item is folded back into its assistant turn."""
        client, forwarded = reasoning_capture_client

        before = len(forwarded)
        response = client.responses.create(
            model=VLLM_MODEL,
            input=[
                {"role": "user", "content": "Pick a number and remember it."},
                {
                    "type": "reasoning",
                    "content": [
                        {"type": "reasoning_text", "text": "I picked 42."}
                    ],
                },
                {"role": "assistant", "content": "Done."},
                {"role": "user", "content": "What number did you pick? /no_think"},
            ],
            reasoning={"effort": "low"},
            tools=[{"type": "function", "name": "lookup", "parameters": {
                "type": "object", "properties": {},
            }}],
            tool_choice="auto",
            temperature=0,
            store=False,
            max_output_tokens=64,
        )
        assert response.status in ("completed", "incomplete"), response.status

        assert response.reasoning.effort == "low"
        assert response.tools[0].type == "function"
        assert response.tools[0].name == "lookup"
        assert response.tool_choice == "auto"

        seen = forwarded[before:]
        assert seen, "backend received no request"
        assert seen[-1]["reasoning_effort"] == "low"
        assert seen[-1]["tools"][0]["function"]["name"] == "lookup"
        assert seen[-1]["tool_choice"] == "auto"
        messages = seen[-1].get("messages")
        assert isinstance(messages, list), seen[-1]
        assistant = next(
            (m for m in messages if m.get("role") == "assistant"), None
        )
        assert assistant is not None, messages
        assert assistant.get("content") == "Done.", assistant
        assert assistant.get("reasoning") == "I picked 42.", assistant

    def test_reasoning_only_stored_continuation(self, reasoning_capture_client):
        client, forwarded = reasoning_capture_client
        first = client.responses.create(
            model=VLLM_MODEL, input="Pick a number.", store=True, stream=False,
        )
        assert len(first.output) == 1
        assert first.output[0].type == "reasoning"
        client.responses.create(
            model=VLLM_MODEL, previous_response_id=first.id,
            input="Now answer.", store=False, stream=False,
        )
        assert len(forwarded) == 2
        assert forwarded[1]["messages"] == [
            {"role": "user", "content": "Pick a number."},
            {"role": "assistant", "content": None, "reasoning": "I picked 42."},
            {"role": "user", "content": "Now answer."},
        ]

    @pytest.mark.parametrize("item", [
        {"type": "reasoning", "encrypted_content": "opaque", "summary": []},
        {"type": "reasoning", "summary": []},
        {"type": "reasoning", "content": "invalid"},
        {"type": "reasoning", "content": [{"type": "reasoning_text", "text": None}]},
    ])
    def test_unreplayable_reasoning_rejected(self, reasoning_capture_client, item):
        client, forwarded = reasoning_capture_client
        with pytest.raises(BadRequestError, match="reasoning input item"):
            client.responses.create(
                model=VLLM_MODEL,
                input=[item, {"role": "user", "content": "Continue."}],
                store=False, stream=False,
            )
        assert not forwarded, "unreplayable reasoning must fail before forwarding"

    @pytest.mark.parametrize("reasoning_capture_client", ["none"], indirect=True)
    @pytest.mark.parametrize("following", [
        [],
        [{"role": "assistant", "content": "Done."}],
        [{"type": "function_call", "call_id": "call_1",
          "name": "lookup", "arguments": "{}"}],
    ])
    def test_reasoning_requires_dialect(self, reasoning_capture_client, following):
        client, forwarded = reasoning_capture_client
        with pytest.raises(BadRequestError, match="a reasoning dialect must be configured"):
            client.responses.create(
                model=VLLM_MODEL,
                input=[{
                    "type": "reasoning",
                    "content": [{"type": "reasoning_text", "text": "I picked 42."}],
                }, *following],
                store=False, stream=False,
            )
        assert not forwarded, "disabled reasoning replay must fail before forwarding"


class TestResponsesCompactionVLLM:
    """Live coverage for automatic context-management compaction."""

    def test_prompt_template_is_rejected_by_combined_request_filter(
        self, compact_client
    ):
        with pytest.raises(BadRequestError) as exc_info:
            compact_client.responses.create(
                model=VLLM_MODEL,
                input="This prompt reference must not reach inference.",
                prompt={"id": "pmpt_compact_rejected"},
                store=False,
            )

        error = exc_info.value
        assert error.status_code == 400
        assert error.type == "invalid_request_error"
        assert error.body["message"] == (
            "prompt templates are supported only for OpenAI-owned upstreams"
        )

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
        assert exc_info.value.status_code == 400, "invalid compaction threshold must return 400"
        assert "compact_threshold" in str(exc_info.value), "error must name the compact_threshold field"

    def test_below_threshold_skips_compaction(self, compact_client):
        first = compact_client.responses.create(
            model=VLLM_MODEL,
            input="Remember the marker BELOW-THRESHOLD-2468. /no_think",
            temperature=0,
            store=True,
            # Qwen3 emits a reasoning block (its /no_think soft switch is not
            # honored through this backend), so budget enough tokens for the
            # reasoning plus the short ack; otherwise the turn truncates to
            # "incomplete" and the continuation rejects the incomplete
            # predecessor.
            max_output_tokens=2048,
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
                    # Comfortably above the reasoning-inflated first-turn history
                    # so this "below threshold" case reliably skips compaction.
                    "compact_threshold": 8000,
                }
            ],
            store=False,
            max_output_tokens=2048,
        )

        assert second.status == "completed"
        assert second.output_text
        assert len(CompactionHandler.requests) == request_count

    def test_usage_less_rehydration_rejects_tokenizer_peak_before_callout(
        self, usage_less_compact_client
    ):
        client, _ = usage_less_compact_client
        conversation = client.conversations.create()
        try:
            first = client.responses.create(
                model=VLLM_MODEL,
                input="abcdefghijklmnopqrstuvwxyz0123456789" * 90,
                conversation=conversation.id,
                store=True,
            )
            assert first.usage is None, "backend intentionally omits usage"
            backend_calls = len(NativeCompactionBackendHandler.requests)
            callouts = len(CompactionHandler.requests)
            persisted_before = len(client.conversations.items.list(conversation.id).data)
            assert persisted_before == 2

            with pytest.raises(APIStatusError) as exc_info:
                client.responses.create(
                    model=VLLM_MODEL,
                    input="continue",
                    conversation=conversation.id,
                    context_management=[{"type": "compaction", "compact_threshold": 1000}],
                    store=True,
                )

            assert exc_info.value.status_code == 413
            assert "during compaction" in str(exc_info.value)
            assert len(NativeCompactionBackendHandler.requests) == backend_calls
            assert len(CompactionHandler.requests) == callouts
            assert len(client.conversations.items.list(conversation.id).data) == persisted_before
        finally:
            client.conversations.delete(conversation.id)

    @requires_real_inference
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

    def test_shared_controls_reach_chat_backend(self, reasoning_capture_client):
        """The SDK request's provider controls survive the checked-in pipeline."""
        client, forwarded = reasoning_capture_client

        response = client.responses.create(
            model=VLLM_MODEL,
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "hello",
                            "prompt_cache_breakpoint": {"mode": "explicit"},
                        },
                        {"type": "input_text", "text": "world"},
                    ],
                }
            ],
            store=False,
            extra_body={
                "safety_identifier": "tenant-user",
                "user": "legacy-user",
                "prompt_cache_key": "cache-key",
                "prompt_cache_retention": "24h",
                "prompt_cache_options": {"ttl": "30m", "mode": "explicit"},
                "text": {"format": {"type": "text"}, "verbosity": "high"},
            },
        )

        assert response.status == "completed"
        assert (
            response.safety_identifier == "tenant-user"
        ), "safety_identifier echoed back"
        assert response.prompt_cache_key == "cache-key", "prompt_cache_key echoed back"
        assert response.text.verbosity == "high", "text.verbosity echoed back"
        assert len(forwarded) == 1, forwarded
        chat = forwarded[0]
        assert chat["messages"][0]["content"] == [
            {
                "type": "text",
                "text": "hello",
                "prompt_cache_breakpoint": {"mode": "explicit"},
            },
            {"type": "text", "text": "world"},
        ], "explicit cache breakpoint must survive content conversion"
        assert (
            chat["safety_identifier"] == "tenant-user"
        ), "safety_identifier must reach the Chat backend"
        assert chat["user"] == "legacy-user", "user must reach the Chat backend"
        assert (
            chat["prompt_cache_key"] == "cache-key"
        ), "prompt_cache_key must reach the Chat backend"
        assert (
            chat["prompt_cache_retention"] == "24h"
        ), "prompt_cache_retention must reach the Chat backend"
        assert chat["prompt_cache_options"] == {
            "ttl": "30m",
            "mode": "explicit",
        }, "prompt_cache_options must reach the Chat backend"
        assert chat["verbosity"] == "high", "text.verbosity must map to Chat verbosity"

    def test_moderation_is_rejected_before_chat_backend(self, reasoning_capture_client):
        client, forwarded = reasoning_capture_client

        with pytest.raises(BadRequestError) as exc_info:
            client.responses.create(
                model=VLLM_MODEL,
                input="hello",
                store=False,
                extra_body={"moderation": {"model": "omni-moderation-latest"}},
            )

        assert (
            exc_info.value.status_code == 400
        ), "moderation must be rejected with 400"
        assert "`moderation` has no Chat Completions representation" in str(
            exc_info.value
        ), "rejection must explain moderation is unrepresentable"
        assert (
            not forwarded
        ), "rejected moderation request must not reach the Chat backend"

    def test_streaming_moderation_is_rejected(self, chat_streaming_client):
        with pytest.raises(BadRequestError) as exc_info:
            chat_streaming_client.responses.create(
                model=VLLM_MODEL,
                input="hello",
                stream=True,
                store=False,
                extra_body={"moderation": {"model": "omni-moderation-latest"}},
            )

        assert (
            exc_info.value.status_code == 400
        ), "streaming moderation must be rejected with 400"
        assert "`moderation` has no Chat Completions representation" in str(
            exc_info.value
        ), "rejection must explain moderation is unrepresentable"

    def test_function_call_output_text_reaches_chat_backend(
        self, reasoning_capture_client
    ):
        """The SDK path lowers text parts to tool text in a real Chat request."""
        client, forwarded = reasoning_capture_client
        for output, expected in [
            ("plain result", "plain result"),
            (
                [
                    {"type": "input_text", "text": "first "},
                    {"type": "input_text", "text": "result"},
                ],
                "first result",
            ),
        ]:
            response = client.responses.create(
                model=VLLM_MODEL,
                input=[
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "lookup",
                        "arguments": "{}",
                    },
                    {
                        "type": "function_call_output",
                        "call_id": "call_1",
                        "output": output,
                    },
                ],
                store=False,
            )
            assert response.status == "completed"
            assert forwarded[-1]["messages"][1] == {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": expected,
            }

        assert len(forwarded) == 2

    def test_function_call_output_unsupported_shapes_stop_before_backend(
        self, reasoning_capture_client
    ):
        """Valid multimodal and invalid scalar outputs cannot become tool text."""
        client, forwarded = reasoning_capture_client
        for output, reason in [
            (
                [{"type": "input_image", "image_url": "data:image/png;base64,AA=="}],
                "input_image",
            ),
            ([{"type": "input_file", "file_id": "file_123"}], "input_file"),
            (42, "must be a string or array of input_text parts"),
            ({"result": 42}, "must be a string or array of input_text parts"),
            (True, "must be a string or array of input_text parts"),
            (None, "must be a string or array of input_text parts"),
        ]:
            with pytest.raises(BadRequestError) as exc_info:
                client.responses.create(
                    model=VLLM_MODEL,
                    input=[
                        {
                            "type": "function_call",
                            "call_id": "call_1",
                            "name": "lookup",
                            "arguments": "{}",
                        },
                        {
                            "type": "function_call_output",
                            "call_id": "call_1",
                            "output": output,
                        },
                    ],
                    store=False,
                )
            assert exc_info.value.status_code == 400, "unsupported output must return 400"
            assert reason in str(exc_info.value), "error message must name the unsupported field"
            assert not forwarded, "unsupported output must not reach the Chat backend"

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

    @pytest.mark.critical_vllm
    @requires_vllm_compat
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

    @pytest.mark.critical_vllm
    @requires_vllm_compat
    def test_structured_output_preserved_with_tools_round_trip(
        self,
        chat_streaming_client,
    ):
        # Regression for issue #1248: the Responses-to-Chat translator used to
        # drop `response_format` whenever tools were translated, silently
        # weakening the structured-output contract. Chat Completions supports
        # both together, so a request pairing `text.format` with a declared tool
        # must still return schema-valid JSON. The tool flows through the
        # translator's tool builder (the exact path that previously deleted the
        # constraint); `tool_choice="none"` keeps the answer direct and
        # deterministic while still exercising that path.
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
        response = chat_streaming_client.responses.create(
            model=VLLM_MODEL,
            input="Return the marker CHAT-JSON-TOOLS-2468. /no_think",
            temperature=0,
            tools=[tool],
            tool_choice="none",
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
            "marker": "CHAT-JSON-TOOLS-2468",
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
            max_output_tokens=512,
        )
        # Experiment, not a proven fix: the 128-token second-turn budget flaked
        # once in CI (SQLite job) while PostgreSQL passed the same commit. The
        # root cause is not yet established -- this turn is free-text
        # (tool_choice="none", no schema) so it is NOT grammar-bounded, and a
        # rehydration/storage-path difference between the backends is not ruled
        # out. Widen only this budget as a controlled experiment and attach
        # diagnostics so the next failure is analyzable: was it a
        # max_output_tokens overrun (incomplete_details.reason / usage), did the
        # model emit a reasoning item (output_types), or was the output empty?
        detail = (
            f"status={second.status!r} "
            f"incomplete_details={getattr(second, 'incomplete_details', None)!r} "
            f"usage={getattr(second, 'usage', None)!r} "
            f"output_types={[item.type for item in second.output]} "
            f"output_text_len={len(second.output_text)}"
        )
        assert second.status == "completed", detail
        assert (
            "68" in second.output_text or "clear" in second.output_text.lower()
        ), detail

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

    @pytest.mark.critical_vllm
    @requires_vllm_compat
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

    @requires_vllm_compat
    def test_backend_error_is_sdk_compatible(self, chat_streaming_client):
        with pytest.raises(NotFoundError) as exc_info:
            chat_streaming_client.responses.create(
                model="model-that-does-not-exist",
                input="This request must fail.",
                store=False,
            )
        assert exc_info.value.status_code == 404, "unknown backend model must surface as 404"

    @requires_vllm_compat
    def test_web_search_streams_terminal_round_as_one_logical_response(
        self, web_search_chat_streaming_client, web_search_chat_streaming_proxy,
    ):
        """Streaming web search resumes through the agentic loop (#986).

        A streaming Responses request is translated to Chat Completions,
        the returned private ``web_search`` tool call is restored to a
        canonical ``web_search_call``, ``openai_web_search`` dispatches the
        query, and inference resumes — all exposed to the client as ONE
        logical Responses SSE lifecycle. The terminal event carries the
        completed web-search item and the final assistant message.
        """
        BraveSearchHandler.reset()

        stream = web_search_chat_streaming_client.responses.create(
            model=VLLM_MODEL,
            input=(
                "You MUST call the web_search tool to look up the latest "
                "Praxis Proxy release, then answer. Do not answer directly. "
                "/no_think"
            ),
            tools=[{"type": "web_search"}],
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

        # AC: one coherent lifecycle framing — created first, completed last,
        # with no intermediate terminal exposed for the internal search round.
        assert event_types[0] == "response.created", event_types
        assert event_types[-1] == "response.completed", event_types
        assert event_types.count("response.created") == 1, event_types
        assert event_types.count("response.completed") == 1, event_types
        assert final_response is not None, (
            "stream must terminate with a response.completed event; "
            f"got: {event_types}"
        )
        assert final_response.status in ("completed", "incomplete"), (
            f"expected completed or incomplete (token limit); got: {final_response.status}"
        )

        # AC: the configured web-search provider is dispatched exactly once.
        assert BraveSearchHandler.request_count == 1, (
            "web search must be dispatched exactly once through the agentic "
            f"loop; got {BraveSearchHandler.request_count} dispatches"
        )

        # AC: the terminal output carries the completed web-search item plus a
        # resumed assistant message — the search round and the final round both
        # surface in the single logical stream.
        output_types = [item.type for item in final_response.output]
        assert "web_search_call" in output_types, (
            "streamed terminal output should contain the completed "
            f"web_search_call; got: {output_types}"
        )
        search_calls = [
            item for item in final_response.output if item.type == "web_search_call"
        ]
        assert all(item.status == "completed" for item in search_calls), (
            f"web_search_call items should be completed; got: {search_calls}"
        )
        assert "message" in output_types, (
            "streamed terminal output should contain the final assistant "
            f"message; got: {output_types}"
        )


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
    return _make_openai_client(proxy_port)


@pytest.fixture(scope="session")
def translated_agentic_proxy(
    tmp_path_factory,
    request,
    mcp_server,
    search_server,
    backend_endpoint,
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
        backend_endpoint=backend_endpoint,
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
    return _make_openai_client(translated_agentic_proxy)


@pytest.fixture(scope="session")
def live_tavily_client(tmp_path_factory, request, backend_endpoint):
    """Run one credentialed Tavily search through the translated vLLM loop."""
    if VLLM_TEST_BACKEND != "live":
        pytest.skip("credentialed Tavily search requires the live backend")

    if not os.environ.get("TAVILY_API_KEY"):
        message = "TAVILY_API_KEY is required for credentialed live web search"
        if REQUIRE_LIVE_WEB_SEARCH:
            pytest.fail(message)
        pytest.skip(message)

    port = _free_port()
    db_dir = tmp_path_factory.mktemp("live-tavily-responses")
    db_path = str(db_dir / "responses.db")
    config_path = _write_agentic_config(
        port,
        db_path,
        0,
        0,
        translate_to_chat=True,
        backend_endpoint=backend_endpoint,
        real_web_search=True,
    )
    binary = _find_binary()

    log_path = str(db_dir / "praxis.log")
    log_file = open(log_path, "w")
    started = False
    client = _make_openai_client(port)
    proc = subprocess.Popen(
        [binary, "-c", config_path],
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    try:
        _wait_for_proxy(port, proc, log_path)
        started = True
        yield client
    finally:
        client.close()
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        log_file.close()
        if not started or request.session.testsfailed > 0:
            with open(log_path) as handle:
                print(
                    f"\n=== Live Tavily Praxis logs ===\n{handle.read()}",
                    file=sys.stderr,
                )
        os.unlink(config_path)


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


class CompatBudgetBackendHandler(BaseHTTPRequestHandler):
    """Return a small private call beside an oversized buffered response."""

    requests: ClassVar[int] = 0

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        type(self).requests += 1
        payload = json.dumps({
            "id": "resp_compat_budget",
            "object": "response",
            "status": "completed",
            "model": "test-model",
            "output": [{
                "type": "function_call", "id": "fc_1", "call_id": "call_1",
                "name": "run_python", "arguments": '{"input":"print(1)"}',
            }],
            "payload": "x" * (256 * 1024),
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt, *args):
        pass


@pytest.fixture()
def client_tool_compat_budget_client(tmp_path, request):
    """SDK proxy with a 64 KiB loop budget and deterministic rich-tool output."""
    CompatBudgetBackendHandler.requests = 0
    backend = HTTPServer(("127.0.0.1", 0), CompatBudgetBackendHandler)
    threading.Thread(target=backend.serve_forever, daemon=True).start()
    port = _free_port()
    config = _load_example_config(
        CLIENT_TOOL_COMPAT_CONFIG_PATH,
        port,
        backend_endpoint=f"127.0.0.1:{backend.server_port}",
        db_path=str(tmp_path / "responses.db"),
    )
    anchor = "                max_infer_iters: 4\n"
    assert config.count(anchor) == 1, "compat example loop configuration changed"
    config = config.replace(anchor, anchor + "                max_retained_bytes: 65536\n", 1)
    config_path = _persist_config(config)
    log_path = str(tmp_path / "praxis.log")
    log_file = open(log_path, "w")
    proc = subprocess.Popen([_find_binary(), "-c", config_path], stdout=log_file, stderr=subprocess.STDOUT)
    started = False
    try:
        _wait_for_proxy(port, proc, log_path)
        started = True
        yield _make_openai_client(port, timeout=60)
    finally:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        log_file.close()
        backend.shutdown()
        backend.server_close()
        if not started or request.session.testsfailed > 0:
            print(f"\n=== Client tool budget Praxis logs ===\n{_read_log_tail(log_path)}", file=sys.stderr)
        os.unlink(config_path)


def test_client_tool_compat_provider_restoration_budget_rejects_through_sdk(client_tool_compat_budget_client):
    """Provider-side restoration overflow is 502 and cannot persist a response."""
    with pytest.raises(APIStatusError) as exc_info:
        client_tool_compat_budget_client.responses.create(
            model="test-model",
            input="x",
            tools=[{"type": "custom", "name": "run_python", "format": {"type": "text"}}],
            store=True,
        )
    assert exc_info.value.status_code == 502
    assert exc_info.value.response.json()["error"]["type"] == "server_error"
    assert CompatBudgetBackendHandler.requests == 1, "inference occurred before restoration failed"
    with pytest.raises(NotFoundError):
        client_tool_compat_budget_client.responses.retrieve("resp_compat_budget")


class TestClientToolCompatVLLM:
    """Issue #1131: rich Codex client tools round-trip through a function-only
    vLLM Responses backend via ``openai_client_tool_compat``.

    The compat filter lowers ``custom``/``namespace``/``shell``/``tool_search``
    declarations to private ``function`` tools on the request (vLLM only ever
    sees functions) and restores the returned ``function_call`` items to their
    canonical typed items on the buffered response — over ``POST /v1/responses``,
    never ``/v1/chat/completions``, and without executing any client tool inside
    Praxis.
    """

    def test_custom_tool_round_trip_lowers_and_restores(
        self, client_tool_compat_client
    ):
        """A ``custom`` client tool is lowered to a private ``function`` vLLM
        accepts; the returned ``function_call`` is restored to a
        ``custom_tool_call`` with the original ``custom`` tool echoed back."""
        response = client_tool_compat_client.responses.create(
            model=VLLM_MODEL,
            input=(
                "You MUST call the apply_patch tool. Do not answer directly. "
                "/no_think"
            ),
            tools=[
                {
                    "type": "custom",
                    "name": "apply_patch",
                    "description": "Apply a unified diff to the workspace.",
                }
            ],
            # Force the call so the small CI model is deterministic; the compat
            # filter lowers this custom selector to a function selector for vLLM.
            tool_choice={"type": "custom", "name": "apply_patch"},
            temperature=0,
            store=False,
            max_output_tokens=256,
        )

        assert response.status == "completed", response
        # Response phase: the function_call is restored to a custom_tool_call.
        custom_calls = [
            item for item in response.output if item.type == "custom_tool_call"
        ]
        assert len(custom_calls) >= 1, (
            "compat filter must restore the function_call to a custom_tool_call; "
            f"got output types: {[i.type for i in response.output]}"
        )
        assert custom_calls[0].name == "apply_patch"
        assert isinstance(custom_calls[0].input, str)
        # No un-restored private function_call may leak to the client.
        assert all(item.type != "function_call" for item in response.output), (
            f"lowered function must not leak: {[i.type for i in response.output]}"
        )
        # Request phase echo: the client sees its original ``custom`` tool back.
        assert any(t.type == "custom" for t in response.tools), response.tools

    def test_single_round_declared_and_discovered_tools_lower_without_leaking(
        self, client_tool_compat_client
    ):
        """General single-request coverage: a declared rich ``custom`` tool and a
        ``tool_search``-discovered ``custom`` tool coexist on one request, are both
        lowered to private ``function`` selectors for the function-only backend, and
        the canonical echo plus restoration stay leak-free.

        A prior client-executed ``tool_search`` discovered ``apply_patch`` while the
        request also declares the rich ``custom`` ``run_python`` and forces the
        discovered tool via ``tool_choice``. The forced discovered selector is
        accepted (lowered ``custom`` -> ``function``, not rejected), and the response
        echoes the *declared* ``run_python`` back as ``custom`` (its echo is not
        clobbered by the discovered set) while never leaking a private ``function``
        tool or ``function_call`` item to the client.

        This asserts only filter-guaranteed, model-independent invariants: whether
        the small CI simulator actually emits the forced call is model-dependent, so
        the test does not require a live tool call. It exercises a single lowering
        only (the example pipeline transitions straight to ``done``), so it passes on
        both pre- and post-fix code and is deliberately NOT the #1249 IRR re-entry
        regression guard — that failure is unreachable through the live agentic loop
        and is pinned synthetically by the Rust unit test
        ``relowering_with_captured_echo_preserves_canonical_restoration``.
        """
        response = client_tool_compat_client.responses.create(
            model=VLLM_MODEL,
            input=[
                {
                    "type": "message",
                    "role": "user",
                    "content": (
                        "You MUST call the apply_patch tool. Do not answer "
                        "directly. /no_think"
                    ),
                },
                {
                    "type": "tool_search_call",
                    "call_id": "call_ts",
                    "execution": "client",
                    "arguments": {"query": "patch"},
                },
                {
                    "type": "tool_search_output",
                    "call_id": "call_ts",
                    "status": "completed",
                    "tools": [
                        {
                            "type": "custom",
                            "name": "apply_patch",
                            "description": "Apply a unified diff to the workspace.",
                            "format": {"type": "text"},
                        }
                    ],
                },
            ],
            tools=[
                {
                    "type": "custom",
                    "name": "run_python",
                    "description": "Run python code in the workspace.",
                }
            ],
            # Force the discovered custom tool: this exercises the discovered
            # tool_choice lowering path (custom -> function selector) and proves the
            # backend accepts it rather than rejecting an undeclared selector.
            # Whether the small CI simulator then honours the forced call is
            # model-dependent, so the assertions below never require a live call.
            tool_choice={"type": "custom", "name": "apply_patch"},
            temperature=0,
            store=False,
            max_output_tokens=256,
        )

        assert response.status == "completed", response
        # No un-restored private ``function_call`` may leak to the client, and any
        # tool call the model did emit must have been restored to a typed
        # ``custom_tool_call`` naming one of the two known tools (never a raw private
        # function call). This holds whether or not the model honoured the forced
        # choice, so it does not depend on the simulator emitting a call.
        assert all(item.type != "function_call" for item in response.output), (
            f"lowered function must not leak: {[i.type for i in response.output]}"
        )
        custom_calls = [
            item for item in response.output if item.type == "custom_tool_call"
        ]
        assert all(
            call.name in {"run_python", "apply_patch"} for call in custom_calls
        ), f"unexpected restored tool name: {[c.name for c in custom_calls]}"
        # The response echoes the *declared* rich tool back as ``custom`` — the
        # discovered set never overwrites the canonical declaration echo, and no
        # lowered private ``function`` tool leaks into the echoed set.
        assert any(
            t.type == "custom" and t.name == "run_python" for t in response.tools
        ), response.tools
        assert all(t.type != "function" for t in response.tools), response.tools
        # The discovered tool was never declared, so it is not echoed.
        assert all(t.name != "apply_patch" for t in response.tools), response.tools

    def test_streaming_custom_tool_restores_lifecycle(
        self, client_tool_compat_client
    ):
        """Issue #1159 (streaming half): a ``custom`` client tool is lowered to a
        private ``function`` vLLM accepts, and the streamed ``function_call``
        lifecycle is restored LIVE to a ``custom_tool_call`` by the
        ``openai_stream_events`` owner — over ``POST /v1/responses`` as one logical
        SSE lifecycle, with the private lowered name never leaking un-restored."""
        stream = client_tool_compat_client.responses.create(
            model=VLLM_MODEL,
            input=(
                "You MUST call the apply_patch tool. Do not answer directly. "
                "/no_think"
            ),
            tools=[
                {
                    "type": "custom",
                    "name": "apply_patch",
                    "description": "Apply a unified diff to the workspace.",
                }
            ],
            # Force the call so the small CI model is deterministic; the compat
            # filter lowers this custom selector to a function selector for vLLM,
            # and openai_stream_events restores the streamed function_call live.
            tool_choice={"type": "custom", "name": "apply_patch"},
            temperature=0,
            store=False,
            stream=True,
            max_output_tokens=256,
        )

        event_types = []
        final_response = None
        for event in stream:
            event_types.append(event.type)
            if event.type == "response.completed":
                final_response = event.response

        # One coherent SSE lifecycle: created first, completed last.
        assert event_types[0] == "response.created", event_types
        assert event_types[-1] == "response.completed", event_types
        assert final_response is not None, (
            f"stream must terminate with a response.completed event; got: {event_types}"
        )
        assert final_response.status == "completed", final_response

        # The streamed function_call is restored LIVE to a custom_tool_call.
        output_types = [item.type for item in final_response.output]
        custom_calls = [
            item for item in final_response.output if item.type == "custom_tool_call"
        ]
        assert len(custom_calls) >= 1, (
            "openai_stream_events must restore the streamed function_call to a "
            f"custom_tool_call; got output types: {output_types}"
        )
        assert custom_calls[0].name == "apply_patch"
        assert isinstance(custom_calls[0].input, str)
        # No un-restored private function_call item may leak to the client.
        assert "function_call" not in output_types, (
            f"lowered function must not leak on the stream: {output_types}"
        )
        # Request-phase echo: the client sees its original ``custom`` tool back.
        assert any(t.type == "custom" for t in final_response.tools), (
            final_response.tools
        )


class TestClientToolCompatChatVLLM:
    """Issue #1206: rich Codex client tools reach a function-only **Chat
    Completions** backend by composing ``openai_client_tool_compat`` with
    ``responses_to_chat_completions`` in one iterative-router step.

    Unlike :class:`TestClientToolCompatVLLM` (native Responses backend), here the
    backend only ever sees ``POST /v1/chat/completions`` with plain ``function``
    tools: compat lowers the rich ``custom``/``namespace``/``shell``/``tool_search``
    declarations into private functions in ``request_body``, r2c translates the
    lowered Responses request into a Chat request, and on the response path r2c
    rebuilds the Responses object first, then compat (buffered) or
    ``openai_stream_events`` (streaming, #1159) restores the private
    ``function_call`` items to their canonical typed items — with no private
    lowered name ever leaking to the client.
    """

    @requires_real_inference
    def test_custom_tool_round_trip_over_chat_backend(
        self, client_tool_compat_chat_client
    ):
        """A ``custom`` client tool is lowered to a private ``function`` the Chat
        backend accepts; the translated ``function_call`` is restored to a
        ``custom_tool_call`` with the original ``custom`` tool echoed back."""
        response = client_tool_compat_chat_client.responses.create(
            model=VLLM_MODEL,
            input=(
                "You MUST call the apply_patch tool. Do not answer directly. "
                "/no_think"
            ),
            tools=[
                {
                    "type": "custom",
                    "name": "apply_patch",
                    "description": "Apply a unified diff to the workspace.",
                }
            ],
            # Force the call so the small CI model is deterministic; compat lowers
            # this custom selector to a function selector for the Chat backend.
            tool_choice={"type": "custom", "name": "apply_patch"},
            temperature=0,
            store=False,
            max_output_tokens=256,
        )

        assert response.status == "completed", response
        custom_calls = [
            item for item in response.output if item.type == "custom_tool_call"
        ]
        assert len(custom_calls) >= 1, (
            "compat must restore the translated function_call to a custom_tool_call "
            f"over a Chat backend; got output types: {[i.type for i in response.output]}"
        )
        assert custom_calls[0].name == "apply_patch"
        assert isinstance(custom_calls[0].input, str)
        # No un-restored private function_call may leak to the client.
        assert all(item.type != "function_call" for item in response.output), (
            f"lowered function must not leak: {[i.type for i in response.output]}"
        )
        # Request-phase echo: the client sees its original ``custom`` tool back.
        assert any(t.type == "custom" for t in response.tools), response.tools

    @requires_real_inference
    def test_streaming_custom_tool_restores_over_chat_backend(
        self, client_tool_compat_chat_client
    ):
        """The streamed Chat tool-call is translated by r2c into a Responses SSE
        lifecycle and restored LIVE to a ``custom_tool_call`` by
        ``openai_stream_events`` — one coherent SSE lifecycle, no leaked name."""
        stream = client_tool_compat_chat_client.responses.create(
            model=VLLM_MODEL,
            input=(
                "You MUST call the apply_patch tool. Do not answer directly. "
                "/no_think"
            ),
            tools=[
                {
                    "type": "custom",
                    "name": "apply_patch",
                    "description": "Apply a unified diff to the workspace.",
                }
            ],
            tool_choice={"type": "custom", "name": "apply_patch"},
            temperature=0,
            store=False,
            stream=True,
            max_output_tokens=256,
        )

        event_types = []
        final_response = None
        for event in stream:
            event_types.append(event.type)
            if event.type == "response.completed":
                final_response = event.response

        assert event_types[0] == "response.created", event_types
        assert event_types[-1] == "response.completed", event_types
        assert final_response is not None, (
            f"stream must terminate with a response.completed event; got: {event_types}"
        )
        assert final_response.status == "completed", final_response

        output_types = [item.type for item in final_response.output]
        custom_calls = [
            item for item in final_response.output if item.type == "custom_tool_call"
        ]
        assert len(custom_calls) >= 1, (
            "openai_stream_events must restore the streamed function_call to a "
            f"custom_tool_call over a Chat backend; got output types: {output_types}"
        )
        assert custom_calls[0].name == "apply_patch"
        assert isinstance(custom_calls[0].input, str)
        # A private ``custom_tool_call_input`` lifecycle must be emitted, never the
        # private ``function_call_arguments`` events for the lowered name.
        assert any(
            evt.startswith("response.custom_tool_call_input") for evt in event_types
        ), event_types
        assert "function_call" not in output_types, (
            f"lowered function must not leak on the stream: {output_types}"
        )
        assert any(t.type == "custom" for t in final_response.tools), (
            final_response.tools
        )

    @requires_real_inference
    def test_custom_tool_output_continuation_over_chat_backend(
        self, client_tool_compat_chat_client
    ):
        """A ``custom_tool_call_output`` re-entered on a stored continuation is
        lowered to a ``function_call_output`` history item and translated by r2c
        into a Chat ``role: tool`` message, so the correlated second turn completes
        without leaking the private lowered name."""
        first = client_tool_compat_chat_client.responses.create(
            model=VLLM_MODEL,
            input=(
                "You MUST call the apply_patch tool. Do not answer directly. "
                "/no_think"
            ),
            tools=[
                {
                    "type": "custom",
                    "name": "apply_patch",
                    "description": "Apply a unified diff to the workspace.",
                }
            ],
            tool_choice={"type": "custom", "name": "apply_patch"},
            temperature=0,
            store=True,
            max_output_tokens=256,
        )

        assert first.status == "completed", first
        custom_calls = [
            item for item in first.output if item.type == "custom_tool_call"
        ]
        assert len(custom_calls) >= 1, (
            f"first turn must produce a custom_tool_call; got: {[i.type for i in first.output]}"
        )
        call_id = custom_calls[0].call_id

        second = client_tool_compat_chat_client.responses.create(
            model=VLLM_MODEL,
            input=[
                {
                    "type": "custom_tool_call_output",
                    "call_id": call_id,
                    "output": "Applied the patch successfully.",
                }
            ],
            tools=[
                {
                    "type": "custom",
                    "name": "apply_patch",
                    "description": "Apply a unified diff to the workspace.",
                }
            ],
            previous_response_id=first.id,
            temperature=0,
            store=True,
            max_output_tokens=256,
        )

        assert second.status == "completed", second
        # The continuation must correlate to the caller's turn and never surface a
        # private lowered function_call/output to the client.
        assert second.previous_response_id == first.id, second.previous_response_id
        assert all(
            item.type not in ("function_call", "function_call_output")
            for item in second.output
        ), f"lowered names must not leak on continuation: {[i.type for i in second.output]}"
        assert any(t.type == "custom" for t in second.tools), second.tools


class RetainedToolSearchBackendHandler(BaseHTTPRequestHandler):
    """Return fixed hosted-search or numeric-heavy responses for budget checks."""

    requests: ClassVar[int] = 0

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        request_body = self.rfile.read(length)
        type(self).requests += 1
        if b"NUMERIC-PARSE-GUARD" in request_body:
            numbers = b",".join([b"1e15"] * 1_024)
            payload = (
                b'{"id":"resp_numeric_budget","object":"response",'
                b'"status":"completed","output":[],"numbers":['
                + numbers
                + b"]}"
            )
        else:
            payload = json.dumps(
                {
                    "id": "resp_tool_search_budget",
                    "object": "response",
                    "status": "completed",
                    "output": [
                        {
                            "type": "tool_search_call",
                            "id": "tsc_budget",
                            "status": "completed",
                            "results": [{"description": "x" * 5_000}],
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


class RetainedFileMetadataHandler(BaseHTTPRequestHandler):
    """Report an oversized file without serving its content."""

    metadata_requests: ClassVar[int] = 0
    content_requests: ClassVar[int] = 0

    def do_GET(self):
        if self.path.endswith("/content"):
            type(self).content_requests += 1
            self.send_response(500)
            self.end_headers()
            return
        type(self).metadata_requests += 1
        payload = json.dumps(
            {"id": "file-budget", "filename": "budget.txt", "content_type": "text/plain", "bytes": 49_152}
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt, *args):
        pass


@pytest.fixture()
def retained_tool_search_client(tmp_path, request):
    """OpenAI SDK client backed by a fixed, oversized hosted search result."""
    RetainedToolSearchBackendHandler.requests = 0
    RetainedFileMetadataHandler.metadata_requests = 0
    RetainedFileMetadataHandler.content_requests = 0
    backend_port = _free_port()
    server = HTTPServer(("127.0.0.1", backend_port), RetainedToolSearchBackendHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    files_port = _free_port()
    files_server = HTTPServer(("127.0.0.1", files_port), RetainedFileMetadataHandler)
    threading.Thread(target=files_server.serve_forever, daemon=True).start()

    proxy_port = _free_port()
    with open("examples/configs/openai/responses/agentic-loop-overflow-fixture.yaml") as f:
        config = f.read()
    legacy_request_filters = (
        "      - filter: openai_responses_format\n"
        "        on_invalid: reject\n"
        "        headers:\n"
        "          format: x-praxis-ai-format\n"
        "          model: x-praxis-ai-model\n"
        "          stream: x-praxis-ai-stream\n\n"
        "      - filter: openai_responses_validate\n"
    )
    if legacy_request_filters not in config:
        raise RuntimeError("retained overflow fixture's request filters changed")
    consolidated_request_filter = (
        "      - filter: openai_responses_request\n"
        "        on_invalid: reject\n"
        "        headers:\n"
        "          format: x-praxis-ai-format\n"
        "          model: x-praxis-ai-model\n"
        "          stream: x-praxis-ai-stream\n"
    )
    config = config.replace(
        legacy_request_filters,
        consolidated_request_filter,
        1,
    )
    config = config.replace("127.0.0.1:8080", f"127.0.0.1:{proxy_port}")
    config = config.replace("127.0.0.1:3001", f"127.0.0.1:{backend_port}")
    # This request uses store=False; omit the fixture's optional SQLite filters
    # so the test runs against the production `full` binary.
    store_filters = (
        "      - filter: openai_response_store\n"
        "        backend: sqlite\n"
        '        database_url: "sqlite://responses.db?mode=rwc"\n'
        "        responses_table: openai_responses\n"
        "        conversations_table: openai_conversations\n\n"
        "      - filter: openai_responses_rehydrate\n\n"
    )
    if store_filters not in config:
        raise RuntimeError("retained overflow fixture's store filters changed")
    config = config.replace(store_filters, "", 1)
    config = config.replace(
        "      - filter: openai_mcp_tool_resolve\n",
        "      - filter: openai_file_resolve\n"
        f'        files_api_url: "http://127.0.0.1:{files_port}"\n'
        "        allow_pre_security_callout: true\n\n"
        "      - filter: openai_mcp_tool_resolve\n",
        1,
    )
    if getattr(request, "param", None) == "doc_extract":
        config = config.replace(
            "      - filter: openai_mcp_tool_resolve\n",
            "      - filter: openai_doc_extract\n"
            "        allow_pre_security_callout: true\n\n"
            "      - filter: openai_mcp_tool_resolve\n",
            1,
        )
    config = config.replace(
        "  allow_private_endpoints: true # example proxies to local backends",
        "  allow_private_endpoints: true # example proxies to local backends\n"
        "  allow_private_upstreams: true # stubbed Files API callouts",
        1,
    )
    if getattr(request, "param", None) == "file_metadata":
        config = config.replace("max_retained_bytes: 16384", "max_retained_bytes: 196608", 1)
    config_path = _persist_config(config)
    log_path = str(tmp_path / "praxis.log")
    log_file = open(log_path, "w")
    started = False
    proc = subprocess.Popen(
        [_find_binary(), "-c", config_path],
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    try:
        _wait_for_proxy(proxy_port, proc, log_path)
        started = True
        yield OpenAI(
            base_url=f"http://127.0.0.1:{proxy_port}/v1",
            api_key="test",
            max_retries=0,
            timeout=60,
        )
    finally:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        log_file.close()
        server.shutdown()
        server.server_close()
        files_server.shutdown()
        files_server.server_close()
        if not started or request.session.testsfailed > 0:
            print(f"\n=== Retained tool search Praxis logs ===\n{_read_log_tail(log_path)}", file=sys.stderr)
        os.unlink(config_path)


@pytest.mark.parametrize(
    "scenario", [
        "direct", "buffered_middle", "empty_stream", "buffered_sse",
        "translated", "header_suppressed", "translated_header_suppressed",
    ]
)
def test_file_budget_failure_after_irr_stream_commit_is_terminal_sse(
    tmp_path, scenario
):
    """Only an actually streamed IRR response changes the later error wire format."""

    class StubHandler(BaseHTTPRequestHandler):
        paths: ClassVar[list[str]] = []
        empty_stream: ClassVar[bool] = scenario == "empty_stream"

        def do_GET(self):
            self.paths.append(self.path)
            if self.path != "/v1/files/file-budget":
                self.send_error(500, "unexpected content fetch")
                return
            payload = json.dumps(
                {"filename": "a.txt", "content_type": "text/plain", "bytes": 49152}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self):
            self.paths.append("POST " + self.path)
            self.rfile.read(int(self.headers["Content-Length"]))
            if scenario.startswith("translated"):
                payload = (
                    b'data: {"id":"chatcmpl-file-budget","object":"chat.completion.chunk",'
                    b'"created":1,"model":"m","choices":[{"index":0,'
                    b'"delta":{"role":"assistant","content":"ok"},"finish_reason":null}]}\n\n'
                    b'data: {"id":"chatcmpl-file-budget","object":"chat.completion.chunk",'
                    b'"created":1,"model":"m","choices":[{"index":0,'
                    b'"delta":{},"finish_reason":"stop"}]}\n\n'
                    b'data: [DONE]\n\n'
                )
            else:
                payload = (
                    b'event: response.created\ndata: {"type":"response.created",'
                    b'"response":{"id":"resp_file_budget","object":"response",'
                    b'"status":"in_progress","output":[]}}\n\n'
                )
            if self.empty_stream:
                payload = b""
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_args):
            pass

    stub = ThreadingHTTPServer(("127.0.0.1", 0), StubHandler)
    thread = threading.Thread(target=stub.serve_forever, daemon=True)
    thread.start()
    proxy_port = _free_port()
    config = f"""
listeners:
  - name: file-budget-stream
    address: "127.0.0.1:{proxy_port}"
    filter_chains: [file-budget-stream]
filter_chains:
  - name: file-budget-stream
    filters:
      - filter: openai_responses_request
        on_invalid: reject
      - filter: iterative_request_router
        initial_step: first
        max_iterations: 2
        steps:
          - name: first
            filters:
              - filter: openai_responses_proxy
              - filter: router
                routes:
                  - path_prefix: "/"
                    cluster: stub
              - filter: load_balancer
                clusters:
                  - name: stub
                    endpoints: ["127.0.0.1:{stub.server_port}"]
            on_result:
              - default: true
                next: resolve
          - name: resolve
            filters:
              - filter: openai_responses_format
              - filter: openai_file_resolve
                files_api_url: "http://127.0.0.1:{stub.server_port}"
                allow_pre_security_callout: true
              - filter: openai_stream_events
              - filter: openai_agentic_loop
                max_infer_iters: 1
                max_retained_bytes: 196608
              - filter: openai_responses_proxy
              - filter: router
                routes:
                  - path_prefix: "/"
                    cluster: stub
              - filter: load_balancer
                clusters:
                  - name: stub
                    endpoints: ["127.0.0.1:{stub.server_port}"]
            on_result:
              - default: true
                done: true
insecure_options:
  allow_private_endpoints: true
  allow_private_upstreams: true
"""
    if scenario.startswith("translated"):
        config = config.replace(
            "              - filter: openai_responses_proxy\n",
            "              - filter: openai_stream_events\n"
            "              - filter: responses_to_chat_completions\n",
            1,
        )
    if scenario.endswith("header_suppressed"):
        first = config.index("          - name: first\n")
        transition = config.index("            on_result:\n", first)
        config = config[:transition] + config[transition:].replace(
            "              - default: true\n                next: resolve",
            "              - status: [200]\n                next: resolve",
            1,
        )
    if scenario in {"buffered_middle", "empty_stream"}:
        config = config.replace("max_iterations: 2", "max_iterations: 3", 1)
        config = config.replace("next: resolve", "next: middle", 1)
        config = config.replace(
            "          - name: resolve\n",
            """          - name: middle
            filters:
              - filter: static_response
                status: 200
                body: '{}'
                headers:
                  - name: Content-Type
                    value: application/json
            on_result:
              - default: true
                next: resolve
          - name: resolve
""",
            1,
        )
    if scenario == "buffered_sse":
        first = config.index("          - name: first\n")
        filters_start = config.index("            filters:\n", first)
        filters_end = config.index("            on_result:\n", filters_start)
        config = (
            config[:filters_start]
            + """            filters:
              - filter: static_response
                status: 200
                body: 'ignored'
                headers:
                  - name: Content-Type
                    value: text/event-stream
"""
            + config[filters_end:]
        )
    config_path = _persist_config(config)
    log_path = tmp_path / "praxis.log"
    with log_path.open("w") as log_file:
        proc = subprocess.Popen(
            [_find_binary(), "-c", config_path],
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        try:
            _wait_for_proxy(proxy_port, proc, str(log_path))
            response = httpx.post(
                f"http://127.0.0.1:{proxy_port}/v1/responses",
                json={
                    "model": "m",
                    "input": [
                        {
                            "role": "user",
                            "content": [{"type": "input_file", "file_id": "file-budget"}],
                        }
                    ],
                    "store": False,
                    "stream": scenario != "buffered_sse",
                },
                timeout=20,
            )
            if scenario in {"buffered_sse", "header_suppressed", "translated_header_suppressed"}:
                assert response.status_code == 502, response.text
                assert response.headers["content-type"].startswith("application/json")
                assert response.json()["error"]["type"] == "server_error"
            else:
                assert response.status_code == 200, response.text
                assert response.headers["content-type"].startswith("text/event-stream")
                frames = [frame for frame in response.text.split("\n\n") if frame]
                event_names = [frame.split("\n", 1)[0] for frame in frames]
                if scenario == "translated":
                    assert "event: response.created" in event_names, response.text
                    assert event_names[-1] == "event: error", response.text
                else:
                    expected = [] if scenario == "empty_stream" else ["event: response.created"]
                    assert event_names == [*expected, "event: error"], response.text
                error = json.loads(frames[-1].split("data: ", 1)[1])
                assert error["type"] == "error"
                assert error["code"] == "server_error"
                assert "agentic retained payload exceeded" in error["message"]
                assert "{\"error\":" not in response.text
            assert StubHandler.paths.count("POST /v1/responses") == (
                0 if scenario == "buffered_sse" else 1
            )
            assert StubHandler.paths.count("/v1/files/file-budget") == 1
            assert "/v1/files/file-budget/content" not in StubHandler.paths
        finally:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            stub.shutdown()
            stub.server_close()
            os.unlink(config_path)
            if proc.returncode not in (0, -2):
                print(_read_log_tail(str(log_path)), file=sys.stderr)


def test_chat_budget_failure_without_logical_finalizer_is_terminal_sdk_sse(tmp_path):
    """A skipped logical finalizer cannot turn a committed Chat stream into truncated 200."""

    class ChatSseHandler(BaseHTTPRequestHandler):
        requests: ClassVar[int] = 0

        def do_POST(self):
            type(self).requests += 1
            self.rfile.read(int(self.headers["Content-Length"]))
            first = (
                b'data: {"id":"chatcmpl-budget","object":"chat.completion.chunk",'
                b'"created":1,"model":"m","choices":[{"index":0,'
                b'"delta":{"role":"assistant","content":"ok"},"finish_reason":null}]}\n\n'
            )
            second = (
                "data: "
                + json.dumps(
                    {
                        "id": "chatcmpl-budget",
                        "object": "chat.completion.chunk",
                        "created": 1,
                        "model": "m",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": "x" * 16_384},
                                "finish_reason": None,
                            }
                        ],
                    }
                )
                + "\n\n"
            ).encode()
            done = b"data: [DONE]\n\n"
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(first) + len(second) + len(done)))
            self.end_headers()
            self.wfile.write(first)
            self.wfile.flush()
            time.sleep(0.1)
            self.wfile.write(second)
            self.wfile.write(done)

        def log_message(self, *_args):
            pass

    backend = ThreadingHTTPServer(("127.0.0.1", 0), ChatSseHandler)
    threading.Thread(target=backend.serve_forever, daemon=True).start()
    proxy_port = _free_port()
    config = f"""
listeners:
  - name: chat-budget-no-finalizer
    address: "127.0.0.1:{proxy_port}"
    filter_chains: [chat-budget-no-finalizer]
filter_chains:
  - name: chat-budget-no-finalizer
    filters:
      - filter: openai_responses_request
        on_invalid: reject
      - filter: iterative_request_router
        initial_step: inference
        max_iterations: 1
        steps:
          - name: inference
            filters:
              - filter: openai_stream_events
                conditions:
                  - when:
                      path_prefix: "/never"
              - filter: openai_agentic_loop
                max_infer_iters: 1
                max_retained_bytes: 65536
                conditions:
                  - when:
                      path_prefix: "/never"
              - filter: responses_to_chat_completions
              - filter: router
                routes:
                  - path_prefix: "/"
                    cluster: stub
              - filter: load_balancer
                clusters:
                  - name: stub
                    endpoints: ["127.0.0.1:{backend.server_port}"]
            on_result:
              - default: true
                done: true
insecure_options:
  allow_private_endpoints: true
"""
    config_path = _persist_config(config)
    log_path = tmp_path / "praxis.log"
    with log_path.open("w") as log_file:
        proc = subprocess.Popen(
            [_find_binary(), "-c", config_path],
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        try:
            _wait_for_proxy(proxy_port, proc, str(log_path))
            client = _make_openai_client(proxy_port)
            events = list(
                client.responses.create(model="m", input="hi", stream=True, store=False)
            )
            event_types = [event.type for event in events]
            assert ChatSseHandler.requests == 1
            assert event_types[0] == "response.created", event_types
            assert event_types[-1] == "error", event_types
            assert "response.completed" not in event_types, event_types
            sequences = [event.sequence_number for event in events]
            assert sequences == list(range(len(sequences))), sequences
        finally:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            backend.shutdown()
            backend.server_close()
            os.unlink(config_path)
            if proc.returncode not in (0, -2):
                print(_read_log_tail(str(log_path)), file=sys.stderr)


class TestAgenticLoopVLLM:
    """Agentic-loop integration tests against the selected backend."""

    def test_initial_retained_budget_rejects_before_sdk_inference(self, agentic_client):
        """The configured loop limit is enforced before request JSON is parsed."""
        with pytest.raises(APIStatusError) as exc_info:
            agentic_client.responses.create(
                model=VLLM_MODEL,
                input="x" * (8 * 1024 * 1024 + 1),
                store=False,
            )

        assert exc_info.value.status_code == 413

    def test_consolidated_request_rejects_raw_body_before_sdk_inference(
        self, retained_tool_search_client
    ):
        """The consolidated request filter rejects before JSON or model dispatch."""
        with pytest.raises(APIStatusError) as exc_info:
            retained_tool_search_client.responses.create(
                model=VLLM_MODEL,
                input="x" * 2_000,
                store=False,
            )

        assert exc_info.value.status_code == 413
        assert RetainedToolSearchBackendHandler.requests == 0

    def test_hosted_tool_search_retained_copies_reject_through_sdk(
        self, retained_tool_search_client
    ):
        with pytest.raises(APIStatusError) as exc_info:
            retained_tool_search_client.responses.create(
                model=VLLM_MODEL,
                input="Find a tool.",
                store=False,
            )

        assert exc_info.value.status_code == 502
        assert "agentic retained payload exceeded" in exc_info.value.response.text
        assert RetainedToolSearchBackendHandler.requests == 1

    def test_numeric_normalization_budget_rejects_through_sdk(
        self, retained_tool_search_client
    ):
        """A short exponent-heavy wire response cannot bypass the parsed-owner bound."""
        with pytest.raises(APIStatusError) as exc_info:
            retained_tool_search_client.responses.create(
                model=VLLM_MODEL,
                input="NUMERIC-PARSE-GUARD",
                store=False,
            )

        assert exc_info.value.status_code == 502
        assert "agentic retained payload exceeded" in exc_info.value.response.text
        assert RetainedToolSearchBackendHandler.requests == 1

    def test_initial_file_resolution_budget_rejects_through_sdk(
        self, retained_tool_search_client
    ):
        """A small budget rejects before staging a Files API transport read."""
        with pytest.raises(APIStatusError) as exc_info:
            retained_tool_search_client.responses.create(
                model=VLLM_MODEL,
                input=[
                    {
                        "role": "user",
                        "content": [{"type": "input_file", "file_id": "file-budget"}],
                    }
                ],
                store=False,
            )

        assert exc_info.value.status_code == 413
        assert "agentic retained payload exceeded" in exc_info.value.response.text
        assert RetainedFileMetadataHandler.metadata_requests == 0
        assert RetainedFileMetadataHandler.content_requests == 0
        assert RetainedToolSearchBackendHandler.requests == 0

    @pytest.mark.parametrize("retained_tool_search_client", ["file_metadata"], indirect=True)
    def test_file_metadata_budget_rejects_before_content_through_sdk(
        self, retained_tool_search_client
    ):
        """A larger budget admits metadata, then blocks oversized content."""
        with pytest.raises(APIStatusError) as exc_info:
            retained_tool_search_client.responses.create(
                model=VLLM_MODEL,
                input=[
                    {
                        "role": "user",
                        "content": [{"type": "input_file", "file_id": "file-budget"}],
                    }
                ],
                store=False,
            )

        assert exc_info.value.status_code == 413
        assert "agentic retained payload exceeded" in exc_info.value.response.text
        assert RetainedFileMetadataHandler.metadata_requests == 1
        assert RetainedFileMetadataHandler.content_requests == 0
        assert RetainedToolSearchBackendHandler.requests == 0

    @pytest.mark.parametrize("retained_tool_search_client", ["doc_extract"], indirect=True)
    def test_document_extraction_budget_rejects_through_sdk(
        self, retained_tool_search_client
    ):
        """Escaped extracted text cannot multiply beyond the shared budget."""
        file_data = "data:text/plain;base64," + base64.b64encode(b"\x01" * 256).decode()
        with pytest.raises(APIStatusError) as exc_info:
            retained_tool_search_client.responses.create(
                model=VLLM_MODEL,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_file",
                                "filename": "controls.txt",
                                "file_data": file_data,
                            }
                        ],
                    }
                ],
                store=False,
            )

        assert exc_info.value.status_code == 413
        assert "during document extraction" in exc_info.value.response.text
        assert RetainedToolSearchBackendHandler.requests == 0

    @requires_real_inference
    def test_explicit_retained_budget_buffered_happy_path(self, agentic_client):
        """The example's explicit 64 MiB aggregate budget admits an ordinary response."""
        response = agentic_client.responses.create(
            model=VLLM_MODEL,
            input="Reply with exactly BUDGET-OK. /no_think",
            store=False,
            max_output_tokens=64,
        )

        assert response.status in ("completed", "incomplete")
        assert response.output

    @requires_real_inference
    def test_explicit_retained_budget_streaming_happy_path(self, agentic_client):
        """The same explicit budget preserves the normal logical SSE lifecycle."""
        stream = agentic_client.responses.create(
            model=VLLM_MODEL,
            input="Reply with exactly STREAM-BUDGET-OK. /no_think",
            store=False,
            stream=True,
            max_output_tokens=64,
        )

        terminal = _assert_stream_contract(_collect_stream(stream))
        assert terminal.status in ("completed", "incomplete")

    def test_mcp_approval_round_trip_executes_once(
        self, agentic_client, agentic_proxy,
    ):
        """An SDK approval response resumes and executes the MCP call once."""
        _, mcp_port, _ = agentic_proxy
        tools = [
            {
                "type": "mcp",
                "server_label": "weather",
                "server_url": f"http://127.0.0.1:{mcp_port}/mcp",
                "allowed_tools": ["get_weather"],
                "require_approval": "always",
            }
        ]
        calls_before = MCPHandler.tool_call_count()

        approval_response = agentic_client.responses.create(
            model=VLLM_MODEL,
            input=(
                "You MUST call the get_weather function for Paris. "
                "Do not answer directly. /no_think"
            ),
            tools=tools,
            store=True,
            max_output_tokens=512,
        )

        approval_requests = [
            item for item in approval_response.output
            if item.type == "mcp_approval_request"
        ]
        assert len(approval_requests) == 1, (
            "approval-gated MCP call should emit exactly one approval request; "
            f"got: {[item.type for item in approval_response.output]}"
        )
        assert MCPHandler.tool_call_count() == calls_before, (
            "the MCP tool must not execute before approval"
        )

        approval = approval_requests[0]
        assert approval.name == "get_weather"
        assert approval.server_label == "weather"

        final_response = agentic_client.responses.create(
            model=VLLM_MODEL,
            previous_response_id=approval_response.id,
            input=[
                {
                    "type": "mcp_approval_response",
                    "approval_request_id": approval.id,
                    "approve": True,
                }
            ],
            tools=tools,
            store=True,
            max_output_tokens=512,
        )

        assert MCPHandler.tool_call_count() == calls_before + 1, (
            "approving the request must execute the MCP tool exactly once"
        )
        output_types = [item.type for item in final_response.output]
        assert "mcp_call" in output_types, (
            f"approved response should contain the MCP result; got: {output_types}"
        )
        # The resume must re-enter inference after dispatch; a small model under
        # /no_think and a 512-token cap may surface only reasoning and no final
        # message, so accept either as proof the tool result fed back into the model.
        assert "message" in output_types or "reasoning" in output_types, (
            f"approved response should resume to model output; got: {output_types}"
        )

        with pytest.raises(APIStatusError) as replay:
            agentic_client.responses.create(
                model=VLLM_MODEL,
                previous_response_id=approval_response.id,
                input=[
                    {
                        "type": "mcp_approval_response",
                        "approval_request_id": approval.id,
                        "approve": True,
                    }
                ],
                tools=tools,
                store=True,
                max_output_tokens=512,
            )
        assert replay.value.status_code == 400
        assert MCPHandler.tool_call_count() == calls_before + 1, (
            "a consumed approval must never execute a second MCP call"
        )

    @requires_vllm_compat
    def test_mcp_approval_resume_streams_without_index_error(
        self, agentic_client, agentic_proxy,
    ):
        """Issue #637 (PR #1029 review): a streamed approval RESUME must
        announce every locally seeded output item before a delta references it.

        The approval resume runs *before* the first inference round. Tool
        discovery (``openai_mcp_tool_resolve``) seeds the ``mcp_list_tools``
        listing at accumulated output index 0 (issue #1022), the executed
        mcp_call lands at index 1, and the resumed model output follows at
        index 2+. Every one of those slots must be announced with a
        ``response.output_item.added`` before a delta references it: if the
        proxy skips index 0 or 1, the OpenAI SDK's streaming accumulator never
        allocates that slot, so a later model item is appended one slot short
        and the next delta indexes past the end of the list, raising
        ``IndexError`` mid-stream -- the exact crash this test guards against.

        Turn 1 is buffered to obtain the approval request id; the RESUME turn
        streams through the SDK's ``responses.stream`` accumulator, which is the
        surface that raised the regression. Reaching ``get_final_response()``
        without an exception is itself the primary assertion.
        """
        _, mcp_port, _ = agentic_proxy
        tools = [
            {
                "type": "mcp",
                "server_label": "weather",
                "server_url": f"http://127.0.0.1:{mcp_port}/mcp",
                "allowed_tools": ["get_weather"],
                "require_approval": "always",
            }
        ]
        calls_before = MCPHandler.tool_call_count()

        approval_response = agentic_client.responses.create(
            model=VLLM_MODEL,
            input=(
                "You MUST call the get_weather function for Paris. "
                "Do not answer directly. /no_think"
            ),
            tools=tools,
            store=True,
            max_output_tokens=512,
        )
        approval_requests = [
            item for item in approval_response.output
            if item.type == "mcp_approval_request"
        ]
        assert len(approval_requests) == 1, (
            "approval-gated MCP call should emit exactly one approval request; "
            f"got: {[item.type for item in approval_response.output]}"
        )
        assert MCPHandler.tool_call_count() == calls_before, (
            "the MCP tool must not execute before approval"
        )
        approval = approval_requests[0]

        # RESUME turn: stream through the SDK accumulator. Building the final
        # snapshot from the event sequence is exactly what raised IndexError
        # before the index-0 mcp_call was announced; letting any such exception
        # propagate fails the test with the regression's own traceback.
        added = []  # (output_index, item_type, item_id)
        text_delta_indices = []
        with agentic_client.responses.stream(
            model=VLLM_MODEL,
            previous_response_id=approval_response.id,
            input=[
                {
                    "type": "mcp_approval_response",
                    "approval_request_id": approval.id,
                    "approve": True,
                }
            ],
            tools=tools,
            store=True,
            max_output_tokens=512,
        ) as stream:
            for event in stream:
                if event.type == "response.output_item.added":
                    item = event.item.model_dump()
                    added.append(
                        (event.output_index, item.get("type"), item.get("id"))
                    )
                elif event.type == "response.output_text.delta":
                    text_delta_indices.append(event.output_index)
            # Replays the accumulated snapshot; raises if any delta referenced
            # an output index that was never announced with output_item.added.
            final_response = stream.get_final_response()

        assert MCPHandler.tool_call_count() == calls_before + 1, (
            "approving the request must execute the MCP tool exactly once"
        )

        # Tool discovery seeds the mcp_list_tools listing at index 0 (issue
        # #1022); it must be announced so the accumulator allocates slot 0 ahead
        # of the resumed tool activity.
        list_added = [a for a in added if a[1] == "mcp_list_tools"]
        assert list_added and list_added[0][0] == 0, (
            "the mcp_list_tools discovery listing must be announced at output "
            f"index 0 ahead of the resumed tool activity; got: {added}"
        )

        # The locally executed mcp_call is announced as exactly one incremental
        # output_item.added at index 1 -- directly after the discovery listing,
        # the slot the accumulator needs before the resumed model output.
        mcp_added = [a for a in added if a[1] == "mcp_call"]
        assert len(mcp_added) == 1, (
            "the resumed mcp_call should surface as exactly one "
            f"response.output_item.added; got: {added}"
        )
        mcp_index = mcp_added[0][0]
        assert mcp_index == 1, (
            "the resumed mcp_call executes before the first inference round but "
            "after tool discovery, so it must be announced at output index 1 "
            f"(behind the mcp_list_tools listing at index 0); got index {mcp_index}"
        )

        # Any resumed model text streams at an output index after the index-1
        # mcp_call -- the ordering the accumulator relies on. (Empty is fine: a
        # small model under /no_think + a 512-token cap may emit only reasoning.)
        assert all(idx > mcp_index for idx in text_delta_indices), (
            "resumed model text must stream at an output index after the "
            f"index-1 mcp_call; mcp_index={mcp_index}, deltas={text_delta_indices}"
        )

        output_types = [item.type for item in final_response.output]
        assert "mcp_call" in output_types, (
            f"resumed response should contain the MCP result; got: {output_types}"
        )

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

    def test_mcp_non_text_content_survives_to_openai_client(
        self,
        agentic_client,
        agentic_proxy,
    ):
        """A non-text MCP tool result reaches the client via the openai SDK.

        The Rust unit tests cover the ``content_blocks_to_output`` transform in
        isolation; this proves the complementary layer the unit test cannot
        reach: a non-text content block (``resource_link``) serializes over the
        wire and is exposed on the ``mcp_call`` output item exactly as the
        OpenAI SDK deserializes the response. The tool *result* is
        server-controlled and deterministic, so only tool *selection* depends
        on the model -- the same reliability profile as
        ``test_mcp_tool_auto_executes_and_returns``.
        """
        _, mcp_port, _ = agentic_proxy
        mcp_url = f"http://127.0.0.1:{mcp_port}/mcp"

        response = agentic_client.responses.create(
            model=VLLM_MODEL,
            input=(
                "You MUST call the get_weather_map function for Paris. "
                "Do not answer directly. /no_think"
            ),
            tools=[
                {
                    "type": "mcp",
                    "server_label": "weather",
                    "server_url": mcp_url,
                    "allowed_tools": ["get_weather_map"],
                    "require_approval": "never",
                }
            ],
            store=False,
            max_output_tokens=512,
        )

        assert response.status in ("completed", "incomplete"), (
            f"expected completed or incomplete (token limit); got: {response.status}"
        )

        mcp_calls = [item.model_dump() for item in response.output if item.type == "mcp_call"]
        assert mcp_calls, (
            "accumulated output should contain the auto-executed MCP tool "
            f"result (mcp_call); got: {[item.type for item in response.output]}"
        )

        # The resource_link block survives serialization end to end: its type
        # and identifying fields land verbatim in the mcp_call output the SDK
        # deserialized, proving non-text MCP content is not flattened or dropped.
        output_text = mcp_calls[0].get("output") or ""
        assert "resource_link" in output_text, (
            f"mcp_call output must carry the resource_link block; got: {output_text}"
        )
        assert "file:///weather/paris-map.png" in output_text, (
            f"resource uri must survive to the client; got: {output_text}"
        )
        assert "paris-weather-map" in output_text, (
            f"resource name must survive to the client; got: {output_text}"
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
            # Approval round trips need durable state to resume, so the proxy
            # requires store=true; store=false is rejected before emit.
            store=True,
            max_output_tokens=256,
        )

        approvals = [
            item for item in response.output if item.type == "mcp_approval_request"
        ]
        assert len(approvals) == 1, response.output
        assert approvals[0].name == "get_weather"
        assert approvals[0].server_label == "weather"
        assert json.loads(approvals[0].arguments).get("city")
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
            # Approval round trips need durable state to resume, so the proxy
            # requires store=true; store=false is rejected before emit.
            store=True,
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
            # Approval round trips need durable state to resume, so the proxy
            # requires store=true; store=false is rejected before emit.
            store=True,
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
        recorded_request_count = len(SimulatorBackendHandler.recorded_requests)
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
            # Live vLLM still needs the hosted call forced for deterministic
            # coverage. The simulator backend is scripted, so use ``auto``
            # there and assert that Praxis preserves it on both rounds.
            tool_choice=(
                "auto"
                if VLLM_TEST_BACKEND == "simulator"
                else {"type": "web_search"}
            ),
            store=False,
            # Room for the continuation round's reasoning plus the final message
            # (Qwen3 emits a reasoning block that /no_think does not suppress).
            max_output_tokens=2048,
        )

        web_search_calls = [
            item for item in response.output if item.type == "web_search_call"
        ]
        assert len(web_search_calls) == 1, response.output
        assert web_search_calls[0].status == "completed"
        assert len(BraveSearchHandler.request_paths) == request_count + 1
        assert any(item.type == "message" for item in response.output)
        if VLLM_TEST_BACKEND == "simulator":
            _assert_simulator_auto_tool_round(
                recorded_request_count,
                tool_name="web_search",
            )

    @requires_real_inference
    def test_live_tavily_web_search_returns_real_sources(
        self,
        live_tavily_client,
    ):
        """A real Tavily call executes inside the vLLM agentic loop."""
        response = live_tavily_client.responses.create(
            model=VLLM_MODEL,
            input=(
                "Use web search exactly once to find the official Rust "
                "programming language website, then report its URL. /no_think"
            ),
            tools=[{"type": "web_search", "search_context_size": "low"}],
            tool_choice={"type": "web_search"},
            include=["web_search_call.action.sources"],
            max_tool_calls=1,
            store=False,
            max_output_tokens=2048,
        )

        calls = [
            item.model_dump()
            for item in response.output
            if item.type == "web_search_call" and item.status == "completed"
        ]
        assert len(calls) == 1, response.output
        sources = calls[0].get("action", {}).get("sources", [])
        assert sources, f"Tavily must return at least one source; got: {calls[0]}"
        urls = [source.get("url") for source in sources]
        assert all(url and url.startswith(("http://", "https://")) for url in urls), (
            f"Tavily sources must contain absolute URLs; got: {sources}"
        )
        assert "https://example.com/mock" not in urls, (
            f"credentialed test must not use the mock Brave result: {sources}"
        )
        assert any(item.type == "message" for item in response.output), response.output

    @requires_vllm_compat
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

    @requires_vllm_compat
    def test_batched_mcp_tools_honor_parallel_tool_calls(
        self,
        agentic_client,
        agentic_proxy,
    ):
        """Two MCP calls emitted in one model round both complete."""
        _, mcp_port, _ = agentic_proxy
        mcp_url = f"http://127.0.0.1:{mcp_port}/mcp"

        response = agentic_client.responses.create(
            model=VLLM_MODEL,
            input=(
                "You MUST call both get_weather and get_time for Paris "
                "in the same turn before answering. Do not omit either "
                "function. /no_think"
            ),
            tools=[
                {
                    "type": "mcp",
                    "server_label": "utilities",
                    "server_url": mcp_url,
                    "allowed_tools": ["get_weather", "get_time"],
                    "require_approval": "never",
                }
            ],
            parallel_tool_calls=True,
            store=False,
            max_output_tokens=512,
        )

        mcp_calls = [item for item in response.output if item.type == "mcp_call"]
        assert len(mcp_calls) == 2, (
            "both calls from the batched model round must execute exactly "
            f"once; got: {[item.type for item in response.output]}"
        )
        assert {item.name for item in mcp_calls} == {"get_weather", "get_time"}

    @requires_vllm_compat
    def test_mcp_approval_batch_resume_executes_multiple_approved_tools(
        self,
        agentic_client,
        agentic_proxy,
    ):
        """Issue #1146: two approval-gated MCP calls emitted in one model
        round, both approved in a SINGLE resume batch, must both execute.

        The pre-fix dispatcher rejected any ``mcp_approval_response`` batch
        with more than one item, so this whole flow returned a 400.
        """
        _, mcp_port, _ = agentic_proxy
        tools = [
            {
                "type": "mcp",
                "server_label": "utilities",
                "server_url": f"http://127.0.0.1:{mcp_port}/mcp",
                "allowed_tools": ["get_weather", "get_time"],
                "require_approval": "always",
            }
        ]
        calls_before = MCPHandler.tool_call_count()

        approval_response = agentic_client.responses.create(
            model=VLLM_MODEL,
            input=(
                "You MUST call both get_weather and get_time for Paris in the "
                "same turn before answering. Do not omit either function. "
                "/no_think"
            ),
            tools=tools,
            parallel_tool_calls=True,
            store=True,
            max_output_tokens=512,
        )

        approval_requests = [
            item for item in approval_response.output
            if item.type == "mcp_approval_request"
        ]
        assert len(approval_requests) == 2, (
            "both approval-gated MCP calls should emit an approval request; "
            f"got: {[item.type for item in approval_response.output]}"
        )
        assert {req.name for req in approval_requests} == {"get_weather", "get_time"}
        assert MCPHandler.tool_call_count() == calls_before, (
            "no MCP tool must execute before approval"
        )

        final_response = agentic_client.responses.create(
            model=VLLM_MODEL,
            previous_response_id=approval_response.id,
            input=[
                {
                    "type": "mcp_approval_response",
                    "approval_request_id": req.id,
                    "approve": True,
                }
                for req in approval_requests
            ],
            tools=tools,
            parallel_tool_calls=True,
            store=True,
            max_output_tokens=512,
        )

        assert MCPHandler.tool_call_count() == calls_before + 2, (
            "approving both requests in one batch must execute both MCP tools"
        )
        mcp_calls = [
            item for item in final_response.output if item.type == "mcp_call"
        ]
        assert {item.name for item in mcp_calls} == {"get_weather", "get_time"}, (
            "approved batch should contain both MCP results; got: "
            f"{[item.type for item in final_response.output]}"
        )

    @requires_vllm_compat
    def test_mcp_approval_batch_resume_mixed_approve_and_deny(
        self,
        agentic_client,
        agentic_proxy,
    ):
        """Issue #1146: a resume batch that approves one call and denies the
        other executes only the approved tool.

        The denied call must not reach the MCP server; only the approved
        ``get_weather`` produces an ``mcp_call`` in the resumed response.
        """
        _, mcp_port, _ = agentic_proxy
        tools = [
            {
                "type": "mcp",
                "server_label": "utilities",
                "server_url": f"http://127.0.0.1:{mcp_port}/mcp",
                "allowed_tools": ["get_weather", "get_time"],
                "require_approval": "always",
            }
        ]
        calls_before = MCPHandler.tool_call_count()

        approval_response = agentic_client.responses.create(
            model=VLLM_MODEL,
            input=(
                "You MUST call both get_weather and get_time for Paris in the "
                "same turn before answering. Do not omit either function. "
                "/no_think"
            ),
            tools=tools,
            parallel_tool_calls=True,
            store=True,
            max_output_tokens=512,
        )

        by_name = {
            item.name: item
            for item in approval_response.output
            if item.type == "mcp_approval_request"
        }
        assert set(by_name) == {"get_weather", "get_time"}, (
            "both approval-gated MCP calls should emit an approval request; "
            f"got: {[item.type for item in approval_response.output]}"
        )

        final_response = agentic_client.responses.create(
            model=VLLM_MODEL,
            previous_response_id=approval_response.id,
            input=[
                {
                    "type": "mcp_approval_response",
                    "approval_request_id": by_name["get_weather"].id,
                    "approve": True,
                },
                {
                    "type": "mcp_approval_response",
                    "approval_request_id": by_name["get_time"].id,
                    "approve": False,
                },
            ],
            tools=tools,
            parallel_tool_calls=True,
            store=True,
            max_output_tokens=512,
        )

        assert MCPHandler.tool_call_count() == calls_before + 1, (
            "only the approved MCP tool must execute; the denied call must not "
            "reach the server"
        )
        mcp_call_names = [
            item.name for item in final_response.output if item.type == "mcp_call"
        ]
        assert mcp_call_names == ["get_weather"], (
            "only the approved call should produce an mcp_call; got: "
            f"{[item.type for item in final_response.output]}"
        )

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

    @requires_vllm_compat
    def test_mcp_tool_streams_local_call_as_incremental_output_items(
        self, agentic_client, agentic_proxy,
    ):
        """Issue #276: locally executed MCP activity is streamed incrementally.

        test_mcp_tool_streams_terminal_round_as_one_logical_response asserts the
        #756 lifecycle framing: the mcp_call the proxy executes locally ends up
        in the terminal response.completed snapshot. That snapshot carries the
        mcp_call with or without #276, so it does not prove the client ever saw
        the tool activity live.

        This sibling asserts the behavior #276 adds: the locally executed
        mcp_call -- which never appears in the model backend's SSE stream -- is
        synthesized as an incremental response.output_item.added /
        response.output_item.done pair, exactly once (no re-emission across IRR
        rounds), before the resumed model output, with an id and output index
        that agree with the terminal snapshot.
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

        # Record incremental output-item events in arrival order, plus every
        # output_text.delta with its output index, so ordering can be asserted
        # against *resumed*-round text without depending on model prose.
        added = []  # (output_index, item_type, item_id, sequence_number)
        done = []   # (output_index, item_type, item_id, sequence_number)
        # #276 tool-specific progress/outcome events: (type, item_id, seq).
        mcp_progress = []
        mcp_added_at = None
        # (position, output_index) per output_text.delta; the ordering check
        # isolates resumed text (output_index > mcp_index) from any round-0
        # narration the model streams before it calls the tool.
        text_deltas = []
        final_response = None

        for position, event in enumerate(stream):
            etype = event.type
            if etype == "response.output_item.added":
                item = event.item.model_dump()
                added.append(
                    (event.output_index, item.get("type"), item.get("id"),
                     event.sequence_number)
                )
                if item.get("type") == "mcp_call" and mcp_added_at is None:
                    mcp_added_at = position
            elif etype == "response.output_item.done":
                item = event.item.model_dump()
                done.append(
                    (event.output_index, item.get("type"), item.get("id"),
                     event.sequence_number)
                )
            elif etype in (
                "response.mcp_call.in_progress",
                "response.mcp_call.completed",
                "response.mcp_call.failed",
            ):
                mcp_progress.append(
                    (etype, event.item_id, event.sequence_number)
                )
            elif etype == "response.output_text.delta":
                text_deltas.append((position, event.output_index))
            elif etype == "response.completed":
                final_response = event.response

        assert final_response is not None, (
            "stream must terminate with a response.completed event"
        )

        # #276 core: the locally executed mcp_call must be streamed as exactly
        # one incremental output_item.added. Without the synthesis it appears
        # only in the terminal snapshot (asserted by the sibling test) and never
        # here -- so this is the assertion that fails when #276 is absent.
        mcp_added = [a for a in added if a[1] == "mcp_call"]
        assert len(mcp_added) == 1, (
            "the locally executed mcp_call should surface as exactly one "
            f"response.output_item.added; got incremental added items: {added}"
        )
        mcp_index, _, mcp_id, mcp_added_seq = mcp_added[0]
        assert mcp_id, f"synthesized mcp_call must carry an id; got: {mcp_added}"

        # Exactly one matching output_item.done for the same id: proves the
        # item is not re-emitted across IRR rounds and that the pair is closed.
        mcp_done = [d for d in done if d[1] == "mcp_call" and d[2] == mcp_id]
        assert len(mcp_done) == 1, (
            "the mcp_call should be closed by exactly one output_item.done for "
            f"id {mcp_id!r}; got incremental done items: {done}"
        )
        assert mcp_done[0][0] == mcp_index, (
            "output_item.done must reuse the added item's output_index; "
            f"added index={mcp_index}, done index={mcp_done[0][0]}"
        )
        assert mcp_added_seq < mcp_done[0][3], (
            "output_item.added must carry a lower sequence_number than its "
            f"output_item.done; added={mcp_added_seq}, done={mcp_done[0][3]}"
        )

        # #276 tool-specific events: between the generic output-item pair the
        # synthesized mcp_call must emit in_progress then completed (the tool
        # succeeds, so never failed). This is what turns an opaque output-item
        # pair into MCP call/result progress the client can render live.
        mcp_prog = [p for p in mcp_progress if p[1] == mcp_id]
        prog_types = [p[0] for p in mcp_prog]
        assert prog_types.count("response.mcp_call.in_progress") == 1, (
            "the synthesized mcp_call must emit exactly one in_progress event "
            f"for id {mcp_id!r}; got progress events: {mcp_progress}"
        )
        assert prog_types.count("response.mcp_call.completed") == 1, (
            "a successful mcp_call must emit exactly one completed outcome "
            f"event for id {mcp_id!r}; got progress events: {mcp_progress}"
        )
        assert "response.mcp_call.failed" not in prog_types, (
            "a successful mcp_call must not emit a failed outcome event; "
            f"got progress events: {mcp_progress}"
        )
        in_progress_seq = next(
            p[2] for p in mcp_prog if p[0] == "response.mcp_call.in_progress"
        )
        completed_seq = next(
            p[2] for p in mcp_prog if p[0] == "response.mcp_call.completed"
        )
        assert (
            mcp_added_seq < in_progress_seq
            < completed_seq
            < mcp_done[0][3]
        ), (
            "mcp_call events must be ordered added -> in_progress -> completed "
            f"-> done by sequence_number; added={mcp_added_seq}, "
            f"in_progress={in_progress_seq}, completed={completed_seq}, "
            f"done={mcp_done[0][3]}"
        )

        # Ordering: synthesized tool activity must precede the *resumed* model
        # output. The model may narrate (stream output_text) in the round that
        # declares the tool, before it is dispatched; that round-0 text occupies
        # an output index below the mcp_call, so it is not "resumed" output.
        # Resumed text is the first output_text.delta whose output_index is
        # above the synthesized mcp_call's index -- the mcp_call added event
        # must precede it.
        first_resumed_text_delta_at = next(
            (pos for pos, output_index in text_deltas if output_index > mcp_index),
            None,
        )
        if first_resumed_text_delta_at is not None:
            assert (
                mcp_added_at is not None
                and mcp_added_at < first_resumed_text_delta_at
            ), (
                "synthesized mcp_call output_item.added must precede the resumed "
                f"model text (output_index > {mcp_index}); "
                f"mcp_added_at={mcp_added_at}, "
                f"first_resumed_text_delta_at={first_resumed_text_delta_at}"
            )

        # Snapshot agrees with the stream: the terminal response.completed
        # carries the same mcp_call (same id) at the same output index it was
        # streamed at -- the incremental events and the final snapshot are one
        # coherent view, not two divergent ones.
        snapshot = [
            (idx, item.type, item.id)
            for idx, item in enumerate(final_response.output)
        ]
        assert any(
            item_type == "mcp_call" and item_id == mcp_id and idx == mcp_index
            for idx, item_type, item_id in snapshot
        ), (
            "the terminal snapshot must agree with the streamed mcp_call "
            f"(id={mcp_id!r}, index={mcp_index}); got snapshot: {snapshot}"
        )

    def test_mcp_discovery_surfaces_mcp_list_tools_output_item(
        self, agentic_client, agentic_proxy,
    ):
        """Issue #1022: successful local MCP discovery surfaces as an output item.

        ``openai_mcp_tool_resolve`` already exposes a *failed* ``mcp_list_tools``
        item (#320). This proves the complementary success path: a resolved
        ``tools/list`` emits exactly one ``mcp_list_tools`` output item per server
        in the buffered terminal response, ahead of the tool activity it enabled,
        with a null error and the discovered tools in ``MCPListToolsTool`` shape.
        Tool *selection* still depends on the model, but the discovery item is
        emitted from the proxy's own resolution and is model-independent.
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

        output = [item.model_dump() for item in response.output]
        list_items = [item for item in output if item.get("type") == "mcp_list_tools"]
        assert len(list_items) == 1, (
            "successful discovery must emit exactly one mcp_list_tools item; "
            f"got output types: {[i.get('type') for i in output]}"
        )
        listing = list_items[0]
        assert listing.get("id", "").startswith("mcpl_"), listing
        assert listing["server_label"] == "weather", listing
        assert listing.get("error") is None, (
            f"a successful discovery listing must carry a null error; got: {listing}"
        )
        tool_names = {tool.get("name") for tool in listing.get("tools", [])}
        assert "get_weather" in tool_names, (
            f"the discovery listing must expose the discovered tool; got: {listing}"
        )
        get_weather = next(
            tool for tool in listing["tools"] if tool.get("name") == "get_weather"
        )
        assert isinstance(get_weather.get("input_schema"), dict), (
            f"each discovered tool must carry an input_schema object; got: {get_weather}"
        )

        # The discovery listing leads the accumulated output, ahead of the tool
        # activity it enabled: its index precedes the first function_call/mcp_call.
        output_types = [item.get("type") for item in output]
        list_index = output_types.index("mcp_list_tools")
        first_tool_index = next(
            (i for i, t in enumerate(output_types) if t in ("function_call", "mcp_call")),
            None,
        )
        assert first_tool_index is not None, (
            f"discovery should precede tool activity; got: {output_types}"
        )
        assert list_index < first_tool_index, (
            "the mcp_list_tools discovery item must precede the tool activity it "
            f"enabled; got: {output_types}"
        )

    def test_mcp_discovery_streams_mcp_list_tools_lifecycle(
        self, agentic_client, agentic_proxy,
    ):
        """Issue #1022: successful discovery streams a full mcp_list_tools lifecycle.

        The buffered sibling above asserts the terminal snapshot carries the
        ``mcp_list_tools`` item; that snapshot alone does not prove the client
        saw discovery live. This asserts the intermediate events #1022 adds: the
        successful listing is synthesized -- ahead of any model output -- as one
        ``output_item.added`` -> ``mcp_list_tools.in_progress`` ->
        ``mcp_list_tools.completed`` -> ``output_item.done`` lifecycle, exactly
        once (no re-emission across IRR rounds), with ids/indices/sequence
        numbers that agree with the terminal snapshot. This is what turns an
        opaque terminal listing into discovery progress the client can render.
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

        added = []  # (position, output_index, item_type, item_id, item, sequence_number)
        done = []   # (output_index, item_type, item_id, sequence_number)
        # (type, item_id, output_index, seq) for the discovery-specific events.
        list_progress = []
        final_response = None

        for position, event in enumerate(stream):
            etype = event.type
            if etype == "response.output_item.added":
                item = event.item.model_dump()
                added.append(
                    (position, event.output_index, item.get("type"), item.get("id"),
                     item, event.sequence_number)
                )
            elif etype == "response.output_item.done":
                item = event.item.model_dump()
                done.append(
                    (event.output_index, item.get("type"), item.get("id"),
                     event.sequence_number)
                )
            elif etype in (
                "response.mcp_list_tools.in_progress",
                "response.mcp_list_tools.completed",
                "response.mcp_list_tools.failed",
            ):
                list_progress.append(
                    (etype, event.item_id, event.output_index, event.sequence_number)
                )
            elif etype == "response.completed":
                final_response = event.response

        assert final_response is not None, (
            "stream must terminate with a response.completed event"
        )

        # #1022 core: the successful discovery surfaces as exactly one incremental
        # output_item.added. Without the synthesis it appears only in the terminal
        # snapshot -- so this is the assertion that fails when #1022 is absent.
        list_added = [a for a in added if a[2] == "mcp_list_tools"]
        assert len(list_added) == 1, (
            "successful discovery should surface as exactly one "
            f"response.output_item.added; got incremental added items: "
            f"{[(a[1], a[2], a[3]) for a in added]}"
        )
        _, list_index, _, list_id, list_item, list_added_seq = list_added[0]
        assert list_id and list_id.startswith("mcpl_"), (
            f"synthesized mcp_list_tools must carry an mcpl_ id; got: {list_id!r}"
        )
        assert list_item.get("server_label") == "weather", list_item
        assert list_item.get("error") is None, (
            f"a successful discovery listing must carry a null error; got: {list_item}"
        )
        assert any(
            tool.get("name") == "get_weather" and isinstance(tool.get("input_schema"), dict)
            for tool in list_item.get("tools", [])
        ), f"the streamed listing must carry the discovered tool: {list_item}"

        # Exactly one matching output_item.done for the same id: proves the
        # listing is not re-emitted across IRR rounds and that the pair is closed.
        list_done = [d for d in done if d[1] == "mcp_list_tools" and d[2] == list_id]
        assert len(list_done) == 1, (
            "the discovery listing should be closed by exactly one "
            f"output_item.done for id {list_id!r}; got done items: {done}"
        )
        assert list_done[0][0] == list_index, (
            "output_item.done must reuse the added item's output_index; "
            f"added index={list_index}, done index={list_done[0][0]}"
        )

        # Between the generic output-item pair the listing must emit in_progress
        # then completed (discovery succeeds, so never failed), all keyed by the
        # same item id and sharing its output index.
        prog = [p for p in list_progress if p[1] == list_id]
        prog_types = [p[0] for p in prog]
        assert prog_types.count("response.mcp_list_tools.in_progress") == 1, (
            "the discovery listing must emit exactly one in_progress event for "
            f"id {list_id!r}; got progress events: {list_progress}"
        )
        assert prog_types.count("response.mcp_list_tools.completed") == 1, (
            "a successful discovery must emit exactly one completed event for "
            f"id {list_id!r}; got progress events: {list_progress}"
        )
        assert "response.mcp_list_tools.failed" not in prog_types, (
            "a successful discovery must not emit a failed event; "
            f"got progress events: {list_progress}"
        )
        for _, _, prog_index, _ in prog:
            assert prog_index == list_index, (
                "discovery progress events must share the listing's output index; "
                f"listing index={list_index}, progress={list_progress}"
            )
        in_progress_seq = next(
            p[3] for p in prog if p[0] == "response.mcp_list_tools.in_progress"
        )
        completed_seq = next(
            p[3] for p in prog if p[0] == "response.mcp_list_tools.completed"
        )
        assert (
            list_added_seq < in_progress_seq < completed_seq < list_done[0][3]
        ), (
            "mcp_list_tools events must be ordered added -> in_progress -> "
            f"completed -> done by sequence_number; added={list_added_seq}, "
            f"in_progress={in_progress_seq}, completed={completed_seq}, "
            f"done={list_done[0][3]}"
        )

        # Discovery precedes all model/tool output: it is the very first
        # output_item.added in the logical stream.
        assert added[0][3] == list_id, (
            "the discovery listing must be the first announced output item, ahead "
            f"of any model output; got added order: {[(a[2], a[3]) for a in added]}"
        )

        # Snapshot agrees with the stream: the terminal response.completed carries
        # the same listing (same id) at the same output index it was streamed at.
        snapshot = [
            (idx, item.type, item.id)
            for idx, item in enumerate(final_response.output)
        ]
        assert any(
            item_type == "mcp_list_tools" and item_id == list_id and idx == list_index
            for idx, item_type, item_id in snapshot
        ), (
            "the terminal snapshot must agree with the streamed mcp_list_tools "
            f"(id={list_id!r}, index={list_index}); got snapshot: {snapshot}"
        )

    @requires_vllm_compat
    def test_web_search_streams_local_call_as_incremental_output_items(
        self, translated_agentic_client,
    ):
        """Issue #276: locally executed web-search activity is streamed live.

        The web-search sibling of
        test_mcp_tool_streams_local_call_as_incremental_output_items. On a
        Chat-Completions-backed model the proxy executes the search locally, so
        the model backend's SSE stream never carries the web_search_call.*
        progress events. Without #276 the search surfaces only in the terminal
        response.completed snapshot; this asserts the behavior #276 adds: the
        locally executed web_search_call is synthesized as an incremental
        output_item.added -> web_search_call.in_progress -> searching ->
        completed -> output_item.done sequence, exactly once (no re-emission
        across IRR rounds), before the resumed model output, with an id and
        output index that agree with the terminal snapshot.
        """
        stream = translated_agentic_client.responses.create(
            model=VLLM_MODEL,
            input=(
                "Search the web for the latest Rust release, then answer. "
                "You MUST use the web search tool. Do not answer directly. "
                "/no_think"
            ),
            tools=[{"type": "web_search_preview"}],
            store=False,
            stream=True,
            max_output_tokens=512,
        )

        # Record incremental output-item events in arrival order, plus every
        # output_text.delta with its output index, so ordering can be asserted
        # against *resumed*-round text without depending on model prose.
        added = []  # (output_index, item_type, item_id, sequence_number)
        done = []   # (output_index, item_type, item_id, sequence_number)
        # #276 tool-specific progress/outcome events: (type, item_id, seq).
        ws_progress = []
        ws_added_at = None
        # (position, output_index) per output_text.delta; the ordering check
        # isolates resumed text (output_index > ws_index) from any round-0
        # narration the model streams before it calls the tool.
        text_deltas = []
        final_response = None

        for position, event in enumerate(stream):
            etype = event.type
            if etype == "response.output_item.added":
                item = event.item.model_dump()
                added.append(
                    (event.output_index, item.get("type"), item.get("id"),
                     event.sequence_number)
                )
                if item.get("type") == "web_search_call" and ws_added_at is None:
                    ws_added_at = position
            elif etype == "response.output_item.done":
                item = event.item.model_dump()
                done.append(
                    (event.output_index, item.get("type"), item.get("id"),
                     event.sequence_number)
                )
            elif etype in (
                "response.web_search_call.in_progress",
                "response.web_search_call.searching",
                "response.web_search_call.completed",
            ):
                ws_progress.append(
                    (etype, event.item_id, event.sequence_number)
                )
            elif etype == "response.output_text.delta":
                text_deltas.append((position, event.output_index))
            elif etype == "response.completed":
                final_response = event.response

        assert final_response is not None, (
            "stream must terminate with a response.completed event"
        )

        # #276 core: the locally executed web_search_call must be streamed as
        # exactly one incremental output_item.added. Without the synthesis it
        # appears only in the terminal snapshot and never here -- so this is the
        # assertion that fails when #276 is absent.
        ws_added = [a for a in added if a[1] == "web_search_call"]
        assert len(ws_added) == 1, (
            "the locally executed web_search_call should surface as exactly one "
            f"response.output_item.added; got incremental added items: {added}"
        )
        ws_index, _, ws_id, ws_added_seq = ws_added[0]
        assert ws_id, (
            f"synthesized web_search_call must carry an id; got: {ws_added}"
        )

        # Exactly one matching output_item.done for the same id: proves the
        # item is not re-emitted across IRR rounds and that the pair is closed.
        ws_done = [d for d in done if d[1] == "web_search_call" and d[2] == ws_id]
        assert len(ws_done) == 1, (
            "the web_search_call should be closed by exactly one "
            f"output_item.done for id {ws_id!r}; got incremental done: {done}"
        )
        assert ws_done[0][0] == ws_index, (
            "output_item.done must reuse the added item's output_index; "
            f"added index={ws_index}, done index={ws_done[0][0]}"
        )
        assert ws_added_seq < ws_done[0][3], (
            "output_item.added must carry a lower sequence_number than its "
            f"output_item.done; added={ws_added_seq}, done={ws_done[0][3]}"
        )

        # #276 tool-specific events: between the generic output-item pair the
        # synthesized web_search_call must emit in_progress -> searching ->
        # completed (the search succeeds; web search has no failed event). This
        # is what turns an opaque output-item pair into search progress the
        # client can render live.
        ws_prog = [p for p in ws_progress if p[1] == ws_id]
        prog_types = [p[0] for p in ws_prog]
        assert prog_types.count("response.web_search_call.in_progress") == 1, (
            "the synthesized web_search_call must emit exactly one in_progress "
            f"event for id {ws_id!r}; got progress events: {ws_progress}"
        )
        assert prog_types.count("response.web_search_call.searching") == 1, (
            "the synthesized web_search_call must emit exactly one searching "
            f"event for id {ws_id!r}; got progress events: {ws_progress}"
        )
        assert prog_types.count("response.web_search_call.completed") == 1, (
            "a successful web_search_call must emit exactly one completed "
            f"outcome event for id {ws_id!r}; got progress events: {ws_progress}"
        )
        in_progress_seq = next(
            p[2] for p in ws_prog
            if p[0] == "response.web_search_call.in_progress"
        )
        searching_seq = next(
            p[2] for p in ws_prog
            if p[0] == "response.web_search_call.searching"
        )
        completed_seq = next(
            p[2] for p in ws_prog
            if p[0] == "response.web_search_call.completed"
        )
        assert (
            ws_added_seq < in_progress_seq
            < searching_seq
            < completed_seq
            < ws_done[0][3]
        ), (
            "web_search_call events must be ordered added -> in_progress -> "
            f"searching -> completed -> done by sequence_number; "
            f"added={ws_added_seq}, in_progress={in_progress_seq}, "
            f"searching={searching_seq}, completed={completed_seq}, "
            f"done={ws_done[0][3]}"
        )

        # Ordering: synthesized tool activity must precede the *resumed* model
        # output. The model may narrate (stream output_text) in the round that
        # declares the tool, before it is dispatched; that round-0 text occupies
        # an output index below the web_search_call, so it is not "resumed"
        # output. Resumed text is the first output_text.delta whose output_index
        # is above the synthesized web_search_call's index -- the web_search_call
        # added event must precede it.
        first_resumed_text_delta_at = next(
            (pos for pos, output_index in text_deltas if output_index > ws_index),
            None,
        )
        if first_resumed_text_delta_at is not None:
            assert (
                ws_added_at is not None
                and ws_added_at < first_resumed_text_delta_at
            ), (
                "synthesized web_search_call output_item.added must precede the "
                f"resumed model text (output_index > {ws_index}); "
                f"ws_added_at={ws_added_at}, "
                f"first_resumed_text_delta_at={first_resumed_text_delta_at}"
            )

        # Snapshot agrees with the stream: the terminal response.completed
        # carries the same web_search_call (same id) at the same output index it
        # was streamed at -- the incremental events and final snapshot are one
        # coherent view, not two divergent ones.
        snapshot = [
            (idx, item.type, item.id)
            for idx, item in enumerate(final_response.output)
        ]
        assert any(
            item_type == "web_search_call" and item_id == ws_id
            and idx == ws_index
            for idx, item_type, item_id in snapshot
        ), (
            "the terminal snapshot must agree with the streamed web_search_call "
            f"(id={ws_id!r}, index={ws_index}); got snapshot: {snapshot}"
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
# Streaming MCP discovery failure (issue #320)
# ---------------------------------------------------------------------------


def _retrieve_with_retry(client, response_id, attempts=15, delay=0.4):
    """Retrieve a stored response, tolerating brief post-stream write latency.

    Streaming persistence completes as the proxy finishes serving the response
    body; a retrieve issued the instant the client's stream iterator returns
    can race that write. Retry a bounded number of times, treating a 404 as
    "not persisted yet" rather than a hard failure.
    """
    from openai import NotFoundError

    last_exc = None
    for _ in range(attempts):
        try:
            return client.responses.retrieve(response_id)
        except NotFoundError as exc:
            last_exc = exc
            time.sleep(delay)
    raise AssertionError(
        f"response {response_id} not retrievable after {attempts} attempts: "
        f"{last_exc}"
    )


def _collect_replay_with_retry(
    client, response_id, *, starting_after=None, attempts=15, delay=0.4
):
    """Replay a stored streaming response's event log via ``GET ?stream=true``.

    The replay event log is flushed as the proxy finishes serving the streamed
    response body; a replay issued the instant the create-stream iterator
    returns can race that write and briefly see the "no replayable event
    stream" 400. Retry a bounded number of times, treating only that specific
    400 as "not flushed yet" and re-raising every other error immediately.
    """
    from openai import BadRequestError

    kwargs: dict[str, Any] = {"stream": True}
    if starting_after is not None:
        kwargs["starting_after"] = starting_after

    last_exc = None
    for _ in range(attempts):
        try:
            return _collect_stream(client.responses.retrieve(response_id, **kwargs))
        except BadRequestError as exc:
            if "replayable" not in str(exc).lower():
                raise
            last_exc = exc
            time.sleep(delay)
    raise AssertionError(
        f"response {response_id} not replayable after {attempts} attempts: "
        f"{last_exc}"
    )


class TestStreamingMcpDiscoveryFailureVLLM:
    """Issue #320: a streaming Responses request whose MCP ``tools/list``
    discovery fails at runtime must surface to the official OpenAI SDK as a
    single, well-formed Responses SSE lifecycle terminating in
    ``response.failed`` -- never an HTTP error, an exception, or a hung
    stream -- and, when ``store`` is effective, the failed resource must be
    retrievable via the SDK.

    The failure is triggered with an MCP ``server_url`` pointing at a dead
    loopback port. The agentic config enables
    ``insecure_options.allow_private_upstreams``, so the loopback MCP callout
    passes SSRF validation and the refused connection is classified as a
    genuine *runtime* discovery failure (-> 200 SSE lifecycle) rather than a
    local SSRF policy rejection (-> HTTP error). Discovery failures
    short-circuit in the request phase, so vLLM is never contacted -- this
    test validates the proxy + SDK streaming contract, not model inference.

    Complements the Rust integration suite, which asserts the raw SSE bytes:
    here we prove the official OpenAI Python SDK parses those frames into a
    clean event stream and a retrievable failed resource.
    """

    def test_streaming_discovery_failure_surfaces_failed_lifecycle(
        self, agentic_client, agentic_proxy,
    ):
        dead_port = _free_port()
        mcp_url = f"http://127.0.0.1:{dead_port}/mcp"

        stream = agentic_client.responses.create(
            model=VLLM_MODEL,
            input="What is the weather in Paris? /no_think",
            tools=[
                {
                    "type": "mcp",
                    "server_label": "weather",
                    "server_url": mcp_url,
                    "allowed_tools": ["get_weather"],
                    "require_approval": "never",
                }
            ],
            store=True,
            stream=True,
            max_output_tokens=128,
            instructions="Be concise",
            metadata={"trace": "sdk-mcp-failure"},
        )

        event_types = []
        response_id = None
        failed_response = None
        for event in stream:
            event_types.append(event.type)
            if event.type == "response.created":
                response_id = event.response.id
            if event.type == "response.failed":
                failed_response = event.response.model_dump()

        # One ordered lifecycle: created first, failed last, with the MCP
        # discovery-failure event in between.
        assert event_types, "the stream must yield at least one event"
        assert event_types[0] == "response.created", event_types
        assert event_types[-1] == "response.failed", event_types
        assert "response.mcp_list_tools.failed" in event_types, event_types

        # The terminal response carries the failure, not a partial success.
        assert failed_response is not None, event_types
        assert failed_response["status"] == "failed", failed_response
        assert failed_response["error"]["code"] == "server_error", (
            failed_response
        )
        assert failed_response["instructions"] == "Be concise", (
            "streamed failure response must echo the request instructions; "
            f"got: {failed_response}"
        )
        assert failed_response["metadata"] == {"trace": "sdk-mcp-failure"}, (
            "streamed failure response must echo the request metadata; "
            f"got: {failed_response}"
        )

        mcp_items = [
            item
            for item in failed_response.get("output", [])
            if item.get("type") == "mcp_list_tools"
        ]
        assert mcp_items, (
            "the failed response must include the mcp_list_tools output item; "
            f"got output types: "
            f"{[i.get('type') for i in failed_response.get('output', [])]}"
        )
        item = mcp_items[0]
        assert item["server_label"] == "weather", item
        assert item["tools"] == [], item
        assert item["error"], "the failed listing item must carry an error"

        # store=True -> the failed resource is persisted and retrievable
        # through the SDK, echoing the same failed status and error.
        assert response_id, "response.created must carry an id for retrieval"
        retrieved = _retrieve_with_retry(agentic_client, response_id)
        rd = retrieved.model_dump()
        assert rd["status"] == "failed", rd
        assert rd["error"]["code"] == "server_error", rd
        assert rd["instructions"] == "Be concise", (
            "retrieved failure response must echo the request instructions; "
            f"got: {rd}"
        )
        assert rd["metadata"] == {"trace": "sdk-mcp-failure"}, (
            "retrieved failure response must echo the request metadata; "
            f"got: {rd}"
        )
        assert any(
            it.get("type") == "mcp_list_tools" for it in rd.get("output", [])
        ), rd


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
      - filter: openai_tool_parse
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
              # Request-phase dispatcher: at request-body EOS on each IRR
              # re-entry it executes the file_search_call items the loop owner
              # assigned in the prior response, reconciling each in place. It
              # never parses the response and never drives the IRR transition
              # (#1046).
              - filter: openai_file_search_callout
                vector_store_url: http://{ogx_endpoint}
                outbound_chain:
                  name: vector-store-outbound
                  filters:
                    - filter: headers
                      request_set:
                        - name: X-Vector-Store-Client
                          value: praxis-ai-gateway
                timeout_ms: 30000
                max_response_bytes: 10485760
                max_total_response_bytes: 67108864
                max_state_bytes: 136314880
                on_failure: closed
                forward_headers:
                  - authorization
              # Sole loop owner: parses each model response, records file-search
              # assignments for the dispatcher, and publishes the single
              # continuation signal (action=loop|done).
              - filter: openai_agentic_loop
                max_infer_iters: 7
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
              - filter: openai_agentic_loop
                key: action
                value: loop
                next: inference
              - default: true
                done: true

insecure_options:
  allow_private_endpoints: true
  # Central SSRF gate for the vector-store callout: permits the loopback OGX
  # endpoint's resolved address at connect time.
  allow_private_upstreams: true
"""


class VectorStoreWitnessHandler(BaseHTTPRequestHandler):
    """Recording shim between the file-search callout and OGX.

    Captures the request headers of every vector-store request the callout
    forwards, then proxies transparently to OGX so the search still runs and
    the full pipeline completes. Tests assert the configured ``outbound_chain``
    actually ran by checking the marker header it injects
    (``X-Vector-Store-Client``) is present on every captured request — proving
    the callout dispatched through the ``FilteredSubrequestExecutor`` outbound
    chain rather than reaching OGX by some other path (or not at all).
    """

    captured_headers: ClassVar[list[dict[str, str]]] = []

    def log_message(self, fmt, *args):
        pass

    def _forward(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b""
        type(self).captured_headers.append(
            {k.lower(): v for k, v in self.headers.items()}
        )
        headers = {
            k: v
            for k, v in self.headers.items()
            if k.lower() not in ("host", "content-length")
        }
        url = f"{OGX_BASE_URL.rstrip('/')}{self.path}"
        with httpx.Client(timeout=300.0) as client:
            with client.stream(
                self.command, url, headers=headers, content=body
            ) as upstream:
                self.send_response(upstream.status_code)
                for key, value in upstream.headers.items():
                    if key.lower() in (
                        "transfer-encoding",
                        "content-length",
                        "connection",
                    ):
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

    def do_DELETE(self):
        self._forward()


def _write_file_search_config(
    praxis_port: int, backend_endpoint: str, ogx_endpoint: str | None = None
) -> str:
    config = FILE_SEARCH_CONFIG_TEMPLATE.format(
        praxis_port=praxis_port,
        ogx_endpoint=ogx_endpoint or _ogx_endpoint(),
        vllm_endpoint=backend_endpoint,
    )
    path = _persist_config(config)
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
def file_search_backend(backend_endpoint):
    """Backend endpoint for the native ``/v1/responses`` file-search path.

    Praxis lowers the hosted tool before the request reaches either the
    simulator or live vLLM, so both modes use the configured backend directly.
    """
    yield backend_endpoint


@pytest.fixture(scope="session")
def file_search_proxy(tmp_path_factory, request, file_search_backend):
    """Start a Praxis proxy with the file-search-callout pipeline.

    The vector-store callout is pointed at an in-process recording shim
    (:class:`VectorStoreWitnessHandler`) that forwards transparently to OGX, so
    a test can assert the configured ``outbound_chain`` ran by inspecting the
    headers the shim captured.
    """
    VectorStoreWitnessHandler.captured_headers = []
    shim_port = _free_port()
    shim = HTTPServer(("127.0.0.1", shim_port), VectorStoreWitnessHandler)
    shim_thread = threading.Thread(target=shim.serve_forever, daemon=True)
    shim_thread.start()

    port = _free_port()
    config_path = _write_file_search_config(
        port,
        file_search_backend,
        ogx_endpoint=f"127.0.0.1:{shim_port}",
    )
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
        shim.shutdown()
        shim_thread.join()
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
    return _make_openai_client(file_search_proxy)


class TestFileSearchVLLM:
    """File search integration tests: vLLM -> Praxis -> OGX -> vLLM."""

    @pytest.mark.critical_vllm
    def test_file_search_with(self, file_search_client, vector_store):
        """vLLM emits function_call(name=file_search) which the proxy
        translates to file_search_call, executes the OGX search callout,
        and returns results to the client.
        """
        store_id, marker = vector_store
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
            # Force the hosted file_search call so the translate/execute path is
            # exercised deterministically instead of depending on the small
            # model to elect the tool.
            tool_choice={"type": "file_search"},
            include=["file_search_call.results"],
            store=False,
            # Room for the continuation round's reasoning plus the final message
            # (Qwen3 emits a reasoning block that /no_think does not suppress).
            max_output_tokens=2048,
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
            assert item.id, "the translated file_search_call must retain a public id"
            assert "call_id" not in item.model_dump(), (
                "the private function call_id must not reach the OpenAI client"
            )

        decoded_results = [
            result
            for item in file_search_items
            for result in (item.results or [])
        ]
        assert decoded_results, "included file_search_call results should be decoded"
        assert any(marker in result.text for result in decoded_results), (
            "decoded file-search results should contain the indexed marker"
        )
        assert all(
            result.file_id and result.filename and result.score is not None
            for result in decoded_results
        ), "decoded file-search results should retain typed result metadata"

        # Prove the callout actually dispatched the vector-store search through
        # the configured FilteredSubrequestExecutor outbound chain — not merely
        # that a file_search_call item surfaced. The recording shim in front of
        # OGX captured each forwarded request; every one must carry the marker
        # header the inline outbound_chain injects, which only the executor path
        # can add.
        captured = VectorStoreWitnessHandler.captured_headers
        assert captured, (
            "the file-search callout must forward at least one vector-store "
            "request through the outbound chain to OGX"
        )
        for headers in captured:
            assert headers.get("x-vector-store-client") == "praxis-ai-gateway", (
                "every vector-store request must carry the outbound_chain marker "
                f"header, proving the callout ran; got headers: {headers}"
            )


# ---------------------------------------------------------------------------
# File search via Chat Completions translation (issue #296)
# ---------------------------------------------------------------------------

FILE_SEARCH_CHAT_CONFIG_PATH = (
    "examples/configs/openai/responses/file-search-chat-completions.yaml"
)


def _write_file_search_chat_config(
    praxis_port: int, backend_endpoint: str
) -> str:
    """Patch the shipped file-search-chat-completions example for testing.

    Exercises the real example config (per repo test requirements) while
    retargeting the vector-store callout at OGX and the model backend at the
    selected /v1/chat/completions endpoint.

    IRR / callout / backend read deadlines are widened to match
    FILE_SEARCH_CONFIG_TEMPLATE: CPU-only vLLM plus OGX is slower when
    the postgres store job co-locates those containers, and a 60s step
    budget can expire before vLLM returns.
    """
    config = _load_example_config(FILE_SEARCH_CHAT_CONFIG_PATH, praxis_port)
    config = config.replace("127.0.0.1:8001", _ogx_endpoint())
    config = config.replace(
        '                  - name: "chat-completions-backend"\n'
        "                    endpoints:\n"
        '                      - "127.0.0.1:3001"',
        f'                  - name: "chat-completions-backend"\n'
        f"                    read_timeout_ms: 300000\n"
        f"                    endpoints:\n"
        f'                      - "{backend_endpoint}"',
    )
    config = config.replace("timeout_ms: 120000", "timeout_ms: 300000")
    config = config.replace("step_timeout_ms: 60000", "step_timeout_ms: 300000")
    config = config.replace("timeout_ms: 5000", "timeout_ms: 30000")
    if f'- "{backend_endpoint}"' not in config:
        raise RuntimeError(
            "file-search-chat-completions.yaml cluster block did not match; "
            "Chat backend endpoint was not patched"
        )

    path = _persist_config(config)
    return path


@pytest.fixture(scope="session")
def file_search_chat_proxy(
    tmp_path_factory, request, backend_endpoint
):
    """Start a Praxis proxy with the file-search Chat Completions pipeline."""
    port = _free_port()
    config_path = _write_file_search_chat_config(port, backend_endpoint)
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
    return _make_openai_client(file_search_chat_proxy)


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
        recorded_request_count = len(SimulatorBackendHandler.recorded_requests)
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

        # The backend-lowered private function must not leak into the echoed
        # request declarations. openai_file_search_callout rewrites
        # request_body.tools into {"type":"function","name":"file_search"} for the
        # Chat backend, but responses_to_chat_completions must echo the hosted
        # tool the client sent, derived from the preserved ResponsesState.tools.
        dumped = response.model_dump()
        echoed_tools = dumped.get("tools") or []
        assert echoed_tools, (
            f"response should echo the client's tool declarations; got {dumped.get('tools')!r}"
        )
        assert any(t.get("type") == "file_search" for t in echoed_tools), (
            f"response.tools must echo the hosted file_search tool; got {echoed_tools}"
        )
        assert not any(
            t.get("type") == "function" and t.get("name") == "file_search"
            for t in echoed_tools
        ), (
            "the backend-only private file_search function must not leak into "
            f"response.tools; got {echoed_tools}"
        )
        echoed_file_search = next(
            t for t in echoed_tools if t.get("type") == "file_search"
        )
        assert store_id in (echoed_file_search.get("vector_store_ids") or []), (
            "the echoed hosted file_search tool must retain the client's "
            f"vector_store_ids; got {echoed_file_search}"
        )
        # The client left tool_choice unset, so the echo must be the hosted
        # default "auto", never the lowered {"type":"function","name":"file_search"}.
        assert dumped.get("tool_choice") == "auto", (
            "response.tool_choice should echo the hosted default 'auto'; got "
            f"{dumped.get('tool_choice')!r}"
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
        if VLLM_TEST_BACKEND == "simulator":
            # The Chat shim scripts this ID; native Responses generates its own.
            assert any(
                item.id == "fc_call_simulator_file_search" for item in file_search_items
            ), "the scripted Chat function call's public id must be preserved"
            _assert_simulator_auto_tool_round(
                recorded_request_count,
                tool_name="file_search",
            )


# ---------------------------------------------------------------------------
# Streaming hosted file search (issue #313)
# ---------------------------------------------------------------------------

FILE_SEARCH_STREAMING_CONFIG_PATH = (
    "examples/configs/openai/responses/file-search-streaming.yaml"
)


def _write_file_search_streaming_config(
    praxis_port: int, backend_endpoint: str
) -> str:
    """Patch the shipped file-search-streaming example for testing.

    Exercises the real #313 streaming example config (per repo test
    requirements) while retargeting the vector-store callout at OGX and the
    model backend at vLLM's native /v1/responses endpoint. The shipped
    example ships tight deadlines suited to a fast provider; CPU-only vLLM
    under co-located CI load (postgres + vLLM + OGX) needs the wider budgets
    already used by the agentic and non-streaming file-search fixtures.
    """
    config = _load_example_config(FILE_SEARCH_STREAMING_CONFIG_PATH, praxis_port)
    config = config.replace("127.0.0.1:8001", _ogx_endpoint())
    # Retarget the model backend and give it a generous read timeout, matching
    # _write_agentic_config. This is also the only occurrence of :3001.
    config = config.replace(
        '- "127.0.0.1:3001"',
        f'- "{backend_endpoint}"\n'
        "                    read_timeout_ms: 300000",
    )
    # Widen the IRR and callout deadlines for slow CPU inference/search.
    config = config.replace("timeout_ms: 120000", "timeout_ms: 300000")
    config = config.replace("step_timeout_ms: 60000", "step_timeout_ms: 300000")
    config = config.replace("timeout_ms: 5000", "timeout_ms: 30000")

    path = _persist_config(config)
    return path


@pytest.fixture(scope="session")
def file_search_streaming_proxy(
    tmp_path_factory, request, file_search_backend
):
    """Start a Praxis proxy with the streaming file-search-callout pipeline."""
    port = _free_port()
    config_path = _write_file_search_streaming_config(
        port, file_search_backend
    )
    binary = _find_binary()

    log_dir = tmp_path_factory.mktemp("file-search-streaming")
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
                    f"\n=== File search streaming proxy logs ===\n{f.read()}",
                    file=sys.stderr,
                )
        os.unlink(config_path)


@pytest.fixture(scope="session")
def file_search_streaming_client(file_search_streaming_proxy):
    """Return an OpenAI client pointed at the streaming file-search proxy."""
    return _make_openai_client(file_search_streaming_proxy)


def _drain_response_stream(stream):
    """Consume a Responses SSE stream.

    Returns the ordered event types, every output item announced via
    output_item.added/.done, and the terminal response status.
    """
    event_types = []
    output_items = []
    terminal_status = None
    for event in stream:
        event_types.append(event.type)
        if event.type in (
            "response.output_item.added",
            "response.output_item.done",
        ):
            output_items.append(event.item)
        if event.type in ("response.completed", "response.incomplete"):
            terminal_status = event.response.status
    return event_types, output_items, terminal_status


class TestFileSearchStreamingVLLM:
    """Issue #313: streaming hosted file_search (stream=True).

    Unlike TestFileSearchVLLM (buffered), this drives the #313 streaming
    example config: openai_stream_events(logical_stream) + openai_file_search_callout
    + openai_responses_proxy (streaming transport auto-derived from
    stream=True). vLLM emits a private
    function_call(name=file_search), which the callout suppresses and replaces
    with a synthesized file_search_call lifecycle, runs the OGX search, and
    streams a terminal re-inference round -- all collapsed onto a single
    client-visible SSE response envelope.
    """

    _INPUT = (
        "Use the file_search tool to find information about the Praxis "
        "marker. Repeat the marker exactly. /no_think"
    )

    @pytest.mark.critical_vllm
    def test_streaming_file_search_lifecycle_events(
        self, file_search_streaming_client, vector_store
    ):
        """The hosted file_search lifecycle is synthesized onto the stream."""
        store_id, _marker = vector_store
        stream = file_search_streaming_client.responses.create(
            model=VLLM_MODEL,
            input=self._INPUT,
            tools=[{"type": "file_search", "vector_store_ids": [store_id]}],
            # Force the hosted file_search call so the synthesized lifecycle is
            # exercised deterministically instead of depending on the small
            # model to elect the tool.
            tool_choice={"type": "file_search"},
            include=["file_search_call.results"],
            store=False,
            stream=True,
            # Room for the continuation round's reasoning plus the final message
            # (Qwen3 emits a reasoning block that /no_think does not suppress).
            max_output_tokens=2048,
        )

        event_types, output_items, terminal_status = _drain_response_stream(
            stream
        )

        assert event_types, "stream should yield at least one event"
        assert event_types[0] == "response.created", event_types
        assert event_types[-1] in (
            "response.completed",
            "response.incomplete",
        ), event_types
        assert terminal_status in ("completed", "incomplete"), terminal_status

        assert "response.file_search_call.completed" in event_types, (
            "streaming hosted file_search must emit the completed lifecycle "
            f"event; got: {event_types}"
        )

        file_search_items = [
            item for item in output_items if item.type == "file_search_call"
        ]
        assert file_search_items, (
            "a hosted file_search_call item must be announced on the stream; "
            f"got event types: {event_types}"
        )
        for item in file_search_items:
            assert item.status in ("searching", "completed", "incomplete"), (
                "file_search_call status should be a known lifecycle state; "
                f"got: {item.status}"
            )

    def test_streaming_file_search_single_logical_stream(
        self, file_search_streaming_client, vector_store
    ):
        """The search round and terminal round collapse into one envelope."""
        store_id, _marker = vector_store
        stream = file_search_streaming_client.responses.create(
            model=VLLM_MODEL,
            input=self._INPUT,
            tools=[{"type": "file_search", "vector_store_ids": [store_id]}],
            include=["file_search_call.results"],
            store=False,
            stream=True,
            max_output_tokens=512,
        )

        event_types, output_items, _status = _drain_response_stream(stream)

        # The #756/#313 logical stream unifies the search round and the
        # terminal re-inference round into ONE client-visible envelope.
        assert event_types.count("response.created") == 1, (
            "multiple model rounds must collapse to a single response.created; "
            f"got: {event_types}"
        )
        terminal_count = sum(
            1
            for t in event_types
            if t in ("response.completed", "response.incomplete")
        )
        assert terminal_count == 1, (
            "the logical stream must emit exactly one terminal response event; "
            f"got: {event_types}"
        )

        # The private function used to drive the search must never surface to
        # the client as a function_call.
        leaked = [
            item for item in output_items if item.type == "function_call"
        ]
        assert not leaked, (
            "the hosted file_search must not leak as a client function_call; "
            f"leaked: {[getattr(item, 'name', '?') for item in leaked]}"
        )


STRUCTURED_OUTPUT_SCHEMA_CASES = [
    pytest.param(
        "Generate a profile for Tom, a software engineer in Raleigh. /no_think",
        {
            "type": "object",
            "properties": {
                "name": {"type": "string", "maxLength": 64},
                "occupation": {"type": "string", "maxLength": 64},
                "city": {"type": "string", "maxLength": 64},
            },
            "required": ["name", "occupation", "city"],
            "additionalProperties": False,
        },
        id="string-types",
    ),
    pytest.param(
        "Generate a profile for Bob, who is 25 years old. /no_think",
        {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name", "age"],
            "additionalProperties": False,
        },
        id="integer-types",
    ),
    pytest.param(
        "Generate an active user named Alice with a verified email. /no_think",
        {
            "type": "object",
            "properties": {
                "username": {"type": "string"},
                "is_active": {"type": "boolean"},
                "email_verified": {"type": "boolean"},
            },
            "required": ["username", "is_active", "email_verified"],
            "additionalProperties": False,
        },
        id="boolean-types",
    ),
    pytest.param(
        "Generate product information for a laptop priced at 999.99. /no_think",
        {
            "type": "object",
            "properties": {
                "product_name": {"type": "string"},
                "price": {"type": "number"},
            },
            "required": ["product_name", "price"],
            "additionalProperties": False,
        },
        id="number-types",
    ),
    pytest.param(
        "Generate a profile for Charlie with Python, JavaScript, and Docker skills. /no_think",
        {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "skills": {
                    "type": "array",
                    "items": {"type": "string", "maxLength": 64},
                    "minItems": 1,
                    "maxItems": 3,
                },
            },
            "required": ["name", "skills"],
            "additionalProperties": False,
        },
        id="array-of-strings",
    ),
    pytest.param(
        'Return exactly this JSON object: {"student_name":"Dana","scores":[85,92,78]}. /no_think',
        {
            "type": "object",
            "properties": {
                "student_name": {"type": "string", "enum": ["Dana"]},
                "scores": {
                    "type": "array",
                    "items": {"type": "integer", "enum": [78, 85, 92]},
                    "minItems": 3,
                    "maxItems": 3,
                },
            },
            "required": ["student_name", "scores"],
            "additionalProperties": False,
        },
        marks=pytest.mark.xfail(
            strict=False,
            raises=json.JSONDecodeError,
            reason="CPU Qwen3-0.6B can exhaust the output limit for this schema",
        ),
        id="array-of-integers",
    ),
    pytest.param(
        "Generate an Engineering team with Alice as lead and Bob as developer. /no_think",
        {
            "type": "object",
            "properties": {
                "team_name": {"type": "string"},
                "members": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "role": {"type": "string"},
                        },
                        "required": ["name", "role"],
                        "additionalProperties": False,
                    },
                    "minItems": 1,
                    "maxItems": 2,
                },
            },
            "required": ["team_name", "members"],
            "additionalProperties": False,
        },
        id="array-of-objects",
    ),
    pytest.param(
        "Generate employee Susan, ID 1001, in Engineering managed by Frank. /no_think",
        {
            "type": "object",
            "properties": {
                "employee": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "employee_id": {"type": "integer"},
                    },
                    "required": ["name", "employee_id"],
                    "additionalProperties": False,
                },
                "department": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "manager": {"type": "string"},
                    },
                    "required": ["name", "manager"],
                    "additionalProperties": False,
                },
            },
            "required": ["employee", "department"],
            "additionalProperties": False,
        },
        id="nested-objects",
    ),
    pytest.param(
        "Generate an active profile for Grace, age 35, salary 120000, with Python and SQL skills, living at 123 Main St in Raleigh, zipcode 27601. /no_think",
        {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "salary": {"type": "number"},
                "is_active": {"type": "boolean"},
                "skills": {
                    "type": "array",
                    "items": {"type": "string", "maxLength": 64},
                    "minItems": 1,
                    "maxItems": 2,
                },
                "address": {
                    "type": "object",
                    "properties": {
                        "street": {"type": "string"},
                        "city": {"type": "string"},
                        "zipcode": {"type": "integer"},
                    },
                    "required": ["street", "city", "zipcode"],
                    "additionalProperties": False,
                },
            },
            "required": ["name", "age", "salary", "is_active", "skills", "address"],
            "additionalProperties": False,
        },
        id="mixed-types-and-structures",
    ),
]


def _assert_matches_schema(value: Any, schema: dict[str, Any], path: str = "$") -> None:
    """Assert the JSON value has the types and closed shape declared by a case."""
    if "enum" in schema:
        assert value in schema["enum"], f"{path} is not an allowed value: {value!r}"
    expected_type = schema["type"]
    if expected_type == "object":
        assert isinstance(value, dict), f"{path} should be an object: {value!r}"
        required = set(schema.get("required", []))
        assert required <= value.keys(), f"{path} is missing {required - value.keys()}"
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            assert value.keys() <= properties.keys(), f"{path} has unexpected keys: {value.keys() - properties.keys()}"
        for key, child_schema in properties.items():
            if key in value:
                _assert_matches_schema(value[key], child_schema, f"{path}.{key}")
        return
    if expected_type == "array":
        assert isinstance(value, list), f"{path} should be an array: {value!r}"
        assert value, f"{path} should not be empty"
        if "minItems" in schema:
            assert len(value) >= schema["minItems"], f"{path} has too few items"
        if "maxItems" in schema:
            assert len(value) <= schema["maxItems"], f"{path} has too many items"
        for index, item in enumerate(value):
            _assert_matches_schema(item, schema["items"], f"{path}[{index}]")
        return
    if expected_type == "string":
        assert isinstance(value, str), f"{path} should be a string: {value!r}"
        if "maxLength" in schema:
            assert len(value) <= schema["maxLength"], f"{path} is too long"
        return
    if expected_type == "integer":
        assert isinstance(value, int) and not isinstance(value, bool), f"{path} should be an integer: {value!r}"
        return
    if expected_type == "number":
        assert isinstance(value, (int, float)) and not isinstance(value, bool), f"{path} should be a number: {value!r}"
        return
    if expected_type == "boolean":
        assert isinstance(value, bool), f"{path} should be a boolean: {value!r}"
        return
    raise AssertionError(f"unsupported test schema type {expected_type!r} at {path}")


@requires_real_inference
@pytest.mark.parametrize("prompt,schema", STRUCTURED_OUTPUT_SCHEMA_CASES)
def test_structured_output_schema_shapes(openai_client, prompt, schema):
    """Exercise nine structured-output schema shapes."""
    text_format = {
        "type": "json_schema",
        "name": "extended_response_shape",
        "description": "A recording-free structured output compatibility case",
        "schema": schema,
        "strict": True,
    }
    response = openai_client.responses.create(
        model=VLLM_MODEL,
        input=prompt,
        stream=False,
        text={"format": text_format},
        temperature=0,
        store=False,
        max_output_tokens=512,
    )

    assert response.text.format.model_dump(exclude_none=True, by_alias=True) == text_format
    _assert_matches_schema(json.loads(response.output_text), schema)


@requires_vllm_compat
def test_include_logprobs_non_streaming(openai_client):
    """Verify the finite include=message.output_text.logprobs scenario."""
    response = openai_client.responses.create(
        model=VLLM_MODEL,
        input="Which planet do humans live on? /no_think",
        stream=False,
        include=["message.output_text.logprobs"],
        store=False,
        max_output_tokens=64,
    )

    messages = [item for item in response.output if item.type == "message"]
    assert len(messages) == 1
    assert messages[0].content[0].logprobs


@requires_vllm_compat
def test_include_logprobs_streaming(openai_client):
    """Verify the streaming include=message.output_text.logprobs scenario."""
    events = list(
        openai_client.responses.create(
            model=VLLM_MODEL,
            input="Which planet do humans live on? /no_think",
            stream=True,
            include=["message.output_text.logprobs"],
            store=False,
            max_output_tokens=64,
        )
    )

    deltas = [event for event in events if event.type == "response.output_text.delta"]
    assert deltas
    assert all(event.logprobs for event in deltas)

    completed = [event for event in events if event.type == "response.completed"]
    assert len(completed) == 1
    messages = [item for item in completed[0].response.output if item.type == "message"]
    assert len(messages) == 1
    assert messages[0].content[0].logprobs


@requires_vllm_compat
def test_response_extra_body_guided_choice(openai_client):
    """Verify the vLLM-specific structured_outputs.choice passthrough case."""
    response = openai_client.responses.create(
        model=VLLM_MODEL,
        input="Classify this sentence: I am feeling really sad today. /no_think",
        stream=False,
        extra_body={"structured_outputs": {"choice": ["joy", "sadness"]}},
        store=False,
        max_output_tokens=16,
    )

    assert response.output_text.strip() in {"joy", "sadness"}


def _create_short_response(openai_client, **options):
    return openai_client.responses.create(
        model=VLLM_MODEL,
        input="Say exactly: RESPONSES-COVERAGE-OK /no_think",
        temperature=0,
        max_output_tokens=64,
        **options,
    )


@pytest.mark.xfail(
    strict=True,
    reason="native Responses does not yet echo prompt_cache_key in streamed response objects",
)
@requires_vllm_compat
def test_openai_response_with_prompt_cache_key_streaming(openai_client):
    """Verify the streaming prompt_cache_key response-shape scenario."""
    cache_key = "responses-coverage-streaming-cache"
    events = list(
        _create_short_response(
            openai_client,
            prompt_cache_key=cache_key,
            stream=True,
            store=False,
        )
    )

    terminal = _assert_stream_contract(events)
    assert events[0].response.prompt_cache_key == cache_key
    assert terminal.prompt_cache_key == cache_key


@pytest.mark.xfail(
    strict=True,
    reason="native Responses does not yet echo prompt_cache_key in finite response objects",
)
@requires_vllm_compat
def test_openai_response_with_prompt_cache_key_and_previous_response(openai_client):
    """Verify the prompt_cache_key plus previous_response_id scenario."""
    cache_key = "responses-coverage-continuation-cache"
    first = _create_short_response(
        openai_client,
        prompt_cache_key=cache_key,
        store=True,
    )
    second = _create_short_response(
        openai_client,
        prompt_cache_key=cache_key,
        previous_response_id=first.id,
        store=False,
    )

    assert first.prompt_cache_key == cache_key
    assert second.prompt_cache_key == cache_key
    assert second.previous_response_id == first.id


@requires_vllm_compat
def test_openai_response_with_truncation_disabled_streaming(openai_client):
    """Verify the streaming truncation response-shape scenario."""
    events = list(
        _create_short_response(
            openai_client,
            truncation="disabled",
            stream=True,
            store=False,
        )
    )

    terminal = _assert_stream_contract(events)
    assert events[0].response.truncation == "disabled"
    assert terminal.truncation == "disabled"


@pytest.mark.xfail(
    strict=True,
    reason="native Responses currently reports the default top_p instead of the requested value",
)
@requires_vllm_compat
def test_openai_response_with_top_p_streaming(openai_client):
    """Verify the streaming top_p response-shape scenario."""
    events = list(
        _create_short_response(
            openai_client,
            top_p=0.8,
            stream=True,
            store=False,
        )
    )

    terminal = _assert_stream_contract(events)
    assert events[0].response.top_p == 0.8
    assert terminal.top_p == 0.8


@pytest.mark.xfail(
    strict=True,
    reason="native Responses currently reports the default top_p instead of the requested value",
)
@requires_vllm_compat
def test_openai_response_with_top_p_and_previous_response(openai_client):
    """Verify the top_p plus previous_response_id scenario."""
    first = _create_short_response(openai_client, top_p=0.7, store=True)
    second = _create_short_response(
        openai_client,
        top_p=0.7,
        previous_response_id=first.id,
        store=False,
    )

    assert first.top_p == 0.7
    assert second.top_p == 0.7
    assert second.previous_response_id == first.id


@requires_vllm_compat
def test_openai_response_with_parallel_tool_calls_disabled_streaming(openai_client):
    """Verify the streaming parallel_tool_calls=false shape scenario."""
    events = list(
        _create_short_response(
            openai_client,
            parallel_tool_calls=False,
            stream=True,
            store=False,
        )
    )

    terminal = _assert_stream_contract(events)
    assert events[0].response.parallel_tool_calls is False
    assert terminal.parallel_tool_calls is False


@requires_vllm_compat
def test_openai_response_with_parallel_tool_calls_and_previous_response(openai_client):
    """Verify the parallel_tool_calls plus continuation scenario."""
    first = _create_short_response(
        openai_client,
        parallel_tool_calls=False,
        store=True,
    )
    second = _create_short_response(
        openai_client,
        parallel_tool_calls=False,
        previous_response_id=first.id,
        store=False,
    )

    assert first.parallel_tool_calls is False
    assert second.parallel_tool_calls is False
    assert second.previous_response_id == first.id


def test_openai_response_with_stream_options_includes_usage(openai_client):
    """Verify the streaming stream_options and usage scenario."""
    events = list(
        _create_short_response(
            openai_client,
            stream=True,
            stream_options={"include_obfuscation": True},
            store=False,
        )
    )

    terminal = _assert_stream_contract(events)
    assert terminal.usage is not None
    assert terminal.usage.total_tokens > 0


def test_openai_response_with_stream_options_non_streaming(openai_client):
    """Verify the finite stream_options acceptance scenario."""
    response = _create_short_response(
        openai_client,
        stream_options={"include_obfuscation": True},
        store=False,
    )

    assert response.object == "response"
    assert response.status == "completed"
    messages = [item for item in response.output if item.type == "message"]
    assert len(messages) == 1
    assert messages[0].content
    assert messages[0].content[0].type == "output_text"
    _assert_usage(response.usage)


def test_openai_response_with_stream_options_and_previous_response(openai_client):
    """Verify the streaming stream_options plus continuation scenario."""
    first = _create_short_response(openai_client, store=True)
    events = list(
        _create_short_response(
            openai_client,
            previous_response_id=first.id,
            stream=True,
            stream_options={"include_obfuscation": True},
            store=False,
        )
    )

    terminal = _assert_stream_contract(events)
    assert terminal.previous_response_id == first.id
    assert terminal.usage is not None


@requires_vllm_compat
def test_invalid_model_raises_not_found_error(openai_client):
    """Verify the SDK exception contract for an unknown model."""
    with pytest.raises(NotFoundError) as exc_info:
        openai_client.responses.create(
            model="nonexistent-model-responses-coverage",
            input="Hello",
        )

    assert exc_info.value.status_code == 404, "unknown model must return 404"


@pytest.mark.xfail(
    strict=True,
    reason="max_tool_calls=0 is not yet rejected at the Responses boundary",
)
def test_invalid_max_tool_calls_raises_bad_request(openai_client):
    """Verify the max_tool_calls lower-bound error scenario."""
    with pytest.raises(BadRequestError) as exc_info:
        openai_client.responses.create(
            model=VLLM_MODEL,
            input="Search for news",
            tools=[{"type": "web_search"}],
            max_tool_calls=0,
        )

    assert exc_info.value.status_code == 400, "max_tool_calls=0 must return 400"
    assert "max_tool_calls" in str(exc_info.value).lower(), "error must name max_tool_calls"


@requires_vllm_compat
def test_invalid_temperature_raises_bad_request(openai_client):
    """Verify propagation of the backend's sampling-temperature validation."""
    with pytest.raises(BadRequestError) as exc_info:
        openai_client.responses.create(
            model=VLLM_MODEL,
            input="Hello",
            temperature=-1.0,
        )

    assert exc_info.value.status_code == 400, "invalid temperature must return 400"
    assert "temperature" in str(exc_info.value).lower(), "error must name temperature"


@requires_vllm_compat
def test_invalid_tool_choice_raises_bad_request(openai_client):
    """Verify the invalid tool_choice error scenario."""
    with pytest.raises(BadRequestError) as exc_info:
        openai_client.responses.create(
            model=VLLM_MODEL,
            input="Hello",
            tools=[
                {
                    "type": "function",
                    "name": "test_tool",
                    "parameters": {"type": "object", "properties": {}},
                }
            ],
            tool_choice="invalid_choice",
        )

    assert exc_info.value.status_code == 400, "invalid tool_choice must return 400"
    assert "tool_choice" in str(exc_info.value).lower(), "error must name tool_choice"


def test_null_tool_choice_succeeds_sdk(chat_streaming_client):
    """Verify null is omitted upstream and normalized to auto for the client."""
    request_start = len(SimulatorBackendHandler.recorded_requests)
    response = chat_streaming_client.responses.create(
        model=VLLM_MODEL,
        input="Hello",
        tools=[
            {
                "type": "function",
                "name": "test_tool",
                "parameters": {"type": "object", "properties": {}},
            }
        ],
        tool_choice=None,
    )
    assert response.status == "completed"
    assert response.tool_choice == "auto"

    if VLLM_TEST_BACKEND == "simulator":
        recorded = [
            body
            for path, body in SimulatorBackendHandler.recorded_requests[request_start:]
            if path.rstrip("/").endswith("/v1/chat/completions")
        ]
        assert len(recorded) == 1, recorded
        translated = recorded[0]
        assert "tool_choice" not in translated, translated
        assert translated["tools"] == [
            {
                "type": "function",
                "function": {
                    "name": "test_tool",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]


@pytest.mark.parametrize(
    "choice_case,tool_choice,with_tools",
    [
        pytest.param("null", None, True, id="null-with-tools"),
        pytest.param("null", None, False, id="null-without-tools"),
        pytest.param("omitted", None, True, id="omitted-with-tools"),
        pytest.param("omitted", None, False, id="omitted-without-tools"),
        pytest.param("value", "none", True, id="none-with-tools"),
        pytest.param("value", "none", False, id="none-without-tools"),
        pytest.param("value", "auto", True, id="auto-with-tools"),
        pytest.param("value", "auto", False, id="auto-without-tools"),
        pytest.param(
            "value",
            {"type": "function", "name": "test_tool"},
            True,
            id="forced-function-with-tools",
        ),
    ],
)
@pytest.mark.parametrize("stream", [False, True])
@requires_vllm_compat
def test_valid_tool_choice_variants_raw_http(
    chat_streaming_client, choice_case, tool_choice, with_tools, stream
):
    """Verify translated tool_choice variants in buffered and streaming modes."""
    body = {
        "model": VLLM_MODEL,
        "input": "Hello",
        "stream": stream,
    }
    if choice_case != "omitted":
        body["tool_choice"] = tool_choice
    if with_tools:
        body["tools"] = [
            {
                "type": "function",
                "name": "test_tool",
                "parameters": {"type": "object", "properties": {}},
            }
        ]

    raw = httpx.post(
        f"{str(chat_streaming_client.base_url).rstrip('/')}/responses",
        headers={"Authorization": "Bearer test"},
        json=body,
        timeout=30,
    )
    assert raw.status_code == 200, (
        f"Failed for case={choice_case}, choice={tool_choice}, "
        f"tools={with_tools}, stream={stream}: {raw.text}"
    )

    expected_choice = "auto" if choice_case in {"null", "omitted"} else tool_choice
    if not stream:
        data = raw.json()
        assert data["tool_choice"] == expected_choice
    else:
        # Check that emitted SSE response objects carry normalized tool_choice
        for line in raw.text.splitlines():
            if line.startswith("data: "):
                try:
                    event = json.loads(line[6:])
                    if isinstance(event, dict) and "response" in event:
                        assert event["response"]["tool_choice"] == expected_choice
                except json.JSONDecodeError:
                    pass


@pytest.mark.parametrize(
    "malformed_choice",
    [
        42,
        {"name": "test_tool"},  # missing "type" discriminator
        "invalid_choice",
    ],
)
@pytest.mark.parametrize("stream", [False, True])
@requires_vllm_compat
def test_malformed_tool_choice_variants_raw_http(
    chat_streaming_client, malformed_choice, stream
):
    """Verify malformed tool_choice variants return 400 over raw HTTP in buffered and streaming modes."""
    raw = httpx.post(
        f"{str(chat_streaming_client.base_url).rstrip('/')}/responses",
        headers={"Authorization": "Bearer test"},
        json={
            "model": VLLM_MODEL,
            "input": "Hello",
            "tools": [
                {
                    "type": "function",
                    "name": "test_tool",
                    "parameters": {"type": "object", "properties": {}},
                }
            ],
            "tool_choice": malformed_choice,
            "stream": stream,
        },
        timeout=30,
    )
    assert raw.status_code == 400, "malformed tool_choice must return 400"
    assert (
        "tool_choice" in raw.text.lower() or "invalid" in raw.text.lower()
    ), "error must identify the invalid tool_choice"


# ---------------------------------------------------------------------------
# openai_file_resolve outbound-chain (fully stubbed upstreams; no vLLM/OGX)
# ---------------------------------------------------------------------------

FILE_RESOLVE_CONFIG_PATH = "examples/configs/openai/responses/file-resolve.yaml"
_FILE_RESOLVE_CONTENT = b"Hello, world!"
_FILE_RESOLVE_B64 = "SGVsbG8sIHdvcmxkIQ=="  # base64("Hello, world!")
_FILE_RESOLVE_ID = "test-file-123"
_FILE_RESOLVE_METADATA = json.dumps(
    {
        "id": _FILE_RESOLVE_ID,
        "object": "file",
        "bytes": len(_FILE_RESOLVE_CONTENT),
        "created_at": 1750000000,
        "filename": "test.txt",
        "purpose": "user_data",
    }
).encode()

# Minimal Responses object the OpenAI SDK can deserialize, so the stubbed
# inference backend can stand in for vLLM. Shape mirrors
# fixtures/inference/recordings/vllm/responses/native-basic-nonstream.json.
_FILE_RESOLVE_STUB_RESPONSE = {
    "id": "resp_file_resolve_stub",
    "object": "response",
    "created_at": 0,
    "model": "gpt-4.1",
    "status": "completed",
    "error": None,
    "incomplete_details": None,
    "instructions": None,
    "max_output_tokens": None,
    "metadata": {},
    "parallel_tool_calls": True,
    "previous_response_id": None,
    "temperature": 1.0,
    "tool_choice": "none",
    "tools": [],
    "top_p": 1.0,
    "output": [
        {
            "id": "msg_file_resolve_stub",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "ok", "annotations": []}],
        }
    ],
    "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
}


class _FilesApiStubHandler(BaseHTTPRequestHandler):
    """Files API stub for `file_id` callouts.

    Answers metadata and content only when the outbound chain stamped
    ``x-file-callout: file-resolve`` on the callout, so a resolved file proves
    the bound outbound pipeline executed.
    """

    callout_headers: ClassVar[list[str | None]] = []

    def do_GET(self):
        header = self.headers.get("x-file-callout")
        type(self).callout_headers.append(header)
        if header != "file-resolve":
            self.send_response(403)
            self.end_headers()
            return
        if self.path == f"/v1/files/{_FILE_RESOLVE_ID}/content":
            body, ctype = _FILE_RESOLVE_CONTENT, "text/plain"
        elif self.path == f"/v1/files/{_FILE_RESOLVE_ID}":
            body, ctype = _FILE_RESOLVE_METADATA, "application/json"
        else:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        pass


class _FileUrlStubHandler(BaseHTTPRequestHandler):
    """Client-controlled `file_url` stub.

    Serves content only when the outbound-chain header is ABSENT, proving the
    credentialed outbound chain never runs for client-supplied URLs.
    """

    requests: ClassVar[int] = 0

    def do_GET(self):
        type(self).requests += 1
        if self.headers.get("x-file-callout") is not None:
            self.send_response(403)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(_FILE_RESOLVE_CONTENT)))
        self.end_headers()
        self.wfile.write(_FILE_RESOLVE_CONTENT)

    def log_message(self, fmt, *args):
        pass


class _FileResolveBackendHandler(BaseHTTPRequestHandler):
    """Capturing inference backend: records the forwarded Responses body."""

    captured: ClassVar[list[dict]] = []

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b""
        try:
            type(self).captured.append(json.loads(body))
        except json.JSONDecodeError:
            pass
        payload = json.dumps(_FILE_RESOLVE_STUB_RESPONSE).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt, *args):
        pass


def _write_file_resolve_config(
    praxis_port: int,
    files_port: int,
    backend_port: int,
    default_port: int,
    file_url_port: int,
) -> str:
    """Patch the shipped file-resolve example for stubbed upstreams."""
    config = _load_example_config(FILE_RESOLVE_CONFIG_PATH, praxis_port)
    # files_api_url and the files-api cluster endpoint both use :9999.
    config = config.replace("127.0.0.1:9999", f"127.0.0.1:{files_port}")
    config = config.replace("127.0.0.1:3001", f"127.0.0.1:{backend_port}")
    config = config.replace("127.0.0.1:3002", f"127.0.0.1:{default_port}")
    # Allow the loopback file_url stub so the client-URL branch resolves
    # without traversing the credentialed outbound chain.
    config = config.replace(
        "        file_url: resolve",
        "        file_url: resolve\n"
        "        allowed_file_url_origins:\n"
        f'          - "http://127.0.0.1:{file_url_port}"',
    )

    return _persist_config(config)


@pytest.fixture()
def file_resolve_stub_env(tmp_path_factory, request):
    """Function-scoped file-resolve proxy with fully stubbed upstreams.

    Needs only the compiled binary — no vLLM and no OGX — so it exercises the
    ``openai_file_resolve`` outbound-chain flow through the OpenAI SDK. Yields
    ``(client, files_stub, backend, file_url)``.
    """
    _FilesApiStubHandler.callout_headers = []
    _FileUrlStubHandler.requests = 0
    _FileResolveBackendHandler.captured = []

    files_port = _free_port()
    backend_port = _free_port()
    default_port = _free_port()
    file_url_port = _free_port()

    servers = [
        HTTPServer(("127.0.0.1", files_port), _FilesApiStubHandler),
        HTTPServer(("127.0.0.1", backend_port), _FileResolveBackendHandler),
        HTTPServer(("127.0.0.1", file_url_port), _FileUrlStubHandler),
    ]
    for server in servers:
        threading.Thread(target=server.serve_forever, daemon=True).start()

    port = _free_port()
    log_dir = tmp_path_factory.mktemp("file-resolve-stub")
    log_path = str(log_dir / "praxis.log")
    log_file = open(log_path, "w")
    config_path = _write_file_resolve_config(
        port, files_port, backend_port, default_port, file_url_port
    )
    started = False
    proc = subprocess.Popen(
        [_find_binary(), "-c", config_path],
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    try:
        _wait_for_proxy(port, proc, log_path)
        started = True
        client = _make_openai_client(port, timeout=60)
        yield (
            client,
            _FilesApiStubHandler,
            _FileResolveBackendHandler,
            f"http://127.0.0.1:{file_url_port}/document.txt",
        )
    finally:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        log_file.close()
        for server in servers:
            server.shutdown()
        if not started or request.session.testsfailed > 0:
            with open(log_path) as f:
                print(
                    f"\n=== File-resolve Praxis logs ===\n{f.read()}",
                    file=sys.stderr,
                )
        os.unlink(config_path)


class TestFileResolveOutboundChain:
    """SDK coverage for the openai_file_resolve outbound-chain callout."""

    def test_file_id_resolved_via_outbound_chain_not_file_url(
        self, file_resolve_stub_env
    ):
        """SDK analogue of the Rust functional test
        ``example_config_outbound_chain_runs_for_file_id_not_file_url``.

        One request carries both a ``file_id`` and a client ``file_url``. The
        Files API stub answers only when the outbound chain stamped
        ``x-file-callout``; the ``file_url`` stub answers only when it did not.
        A single completed response with both parts inlined therefore proves the
        credentialed outbound chain ran for the ``file_id`` callout but never for
        the client-controlled ``file_url`` download.
        """
        client, files_stub, backend, file_url = file_resolve_stub_env

        response = client.responses.create(
            model="gpt-4.1",
            input=[
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_file", "file_id": _FILE_RESOLVE_ID},
                        {"type": "input_file", "file_url": file_url},
                        {"type": "input_text", "text": "summarize"},
                    ],
                }
            ],
            store=False,
        )

        assert response.status == "completed"
        assert len(backend.captured) == 1, backend.captured
        content = backend.captured[0]["input"][0]["content"]

        # file_id inlined as raw base64 file_data (the outbound chain ran).
        assert content[0]["file_data"] == _FILE_RESOLVE_B64, content
        assert "file_id" not in content[0], content

        # file_url inlined as a data URI (the outbound chain did NOT run for it).
        assert content[1]["file_data"] == (
            f"data:text/plain;base64,{_FILE_RESOLVE_B64}"
        ), content

        # Every Files API callout carried the outbound-chain marker header.
        assert files_stub.callout_headers, "Files API stub should receive callouts"
        assert all(h == "file-resolve" for h in files_stub.callout_headers), (
            files_stub.callout_headers
        )

    def test_repeated_file_references_reuse_source_specific_cache(
        self, file_resolve_stub_env
    ):
        client, files_stub, backend, file_url = file_resolve_stub_env

        response = client.responses.create(
            model="gpt-4.1",
            input=[
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_file", "file_id": _FILE_RESOLVE_ID},
                        {"type": "input_file", "file_id": _FILE_RESOLVE_ID},
                        {"type": "input_file", "file_url": file_url},
                        {"type": "input_file", "file_url": file_url},
                    ],
                }
            ],
            store=False,
        )

        assert response.status == "completed", "file resolution should complete"
        assert len(backend.captured) == 1, "one Responses request should reach the backend"
        content = backend.captured[0]["input"][0]["content"]
        assert [part["file_data"] for part in content] == [
            _FILE_RESOLVE_B64,
            _FILE_RESOLVE_B64,
            f"data:text/plain;base64,{_FILE_RESOLVE_B64}",
            f"data:text/plain;base64,{_FILE_RESOLVE_B64}",
        ], "repeated file IDs and URLs should inline in input order"
        assert len(files_stub.callout_headers) == 2, (
            "one metadata and one content fetch should serve both file_id parts"
        )
        assert _FileUrlStubHandler.requests == 1, (
            "repeated file_url parts should share one fetch"
        )

    def test_file_url_rejects_cloud_metadata(self, file_resolve_stub_env):
        """Private-origin opt-ins must not grant access to metadata services."""
        client, _, backend, _ = file_resolve_stub_env

        with pytest.raises(PermissionDeniedError) as exc_info:
            client.responses.create(
                model="gpt-4.1",
                input=[
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {
                                "type": "input_file",
                                "file_url": (
                                    "http://169.254.169.254/latest/meta-data/"
                                ),
                            }
                        ],
                    }
                ],
                store=False,
            )

        assert exc_info.value.status_code == 403, "metadata file URLs should return 403"
        assert "blocked by security policy" in str(
            exc_info.value
        ), "metadata file URLs should report the security-policy rejection"
        assert backend.captured == [], "blocked metadata requests must not reach the backend"


# ---------------------------------------------------------------------------
# Model rewrite on Chat Completions
# ---------------------------------------------------------------------------

MODEL_REWRITE_CONFIG_PATH = "examples/configs/openai/responses/model-rewrite.yaml"

# The client-facing name the alias table maps onto the real backend model.
# Matches the shipped example's "codex-*" wildcard alias.
MODEL_REWRITE_CLIENT_MODEL = "codex-mini-2026-06-24"


def _write_model_rewrite_config(praxis_port: int, backend_endpoint: str) -> str:
    """Patch the shipped model-rewrite example for the selected backend.

    Substituting the backend model name for the example's `llama-3.3-70b`
    updates the alias target, the `default_model`, and the router's
    `x-praxis-ai-effective-model` match in one pass, so the rewritten model
    is both what the backend receives and what selects the cluster. All
    three example clusters point at the one backend under test.
    """
    config = _load_example_config(MODEL_REWRITE_CONFIG_PATH, praxis_port)
    for placeholder in ("127.0.0.1:3001", "127.0.0.1:3002", "127.0.0.1:3003"):
        config = config.replace(placeholder, backend_endpoint)
    config = config.replace("llama-3.3-70b", VLLM_MODEL)

    return _persist_config(config)


@pytest.fixture(scope="session")
def model_rewrite_proxy(tmp_path_factory, request, backend_endpoint):
    """Start a Praxis proxy with the shipped model-rewrite pipeline."""
    port = _free_port()
    config_path = _write_model_rewrite_config(port, backend_endpoint)
    binary = _find_binary()

    log_dir = tmp_path_factory.mktemp("model-rewrite")
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
                    f"\n=== Model rewrite proxy logs ===\n{f.read()}",
                    file=sys.stderr,
                )
        os.unlink(config_path)


@pytest.fixture(scope="session")
def model_rewrite_client(model_rewrite_proxy):
    """Return an OpenAI client pointed at the model-rewrite proxy."""
    return _make_openai_client(model_rewrite_proxy)


class TestModelRewriteChatCompletionsVLLM:
    """openai_responses_model_rewrite applied to POST /v1/chat/completions.

    A gateway advertises one client-facing model name while the selected
    backend requires its own. The filter rewrites the top-level `model`
    before the request leaves the proxy, so the backend never sees the
    client-facing name and never 404s on an unknown model.
    """

    def test_chat_completions_alias_rewritten_for_backend(
        self, model_rewrite_client
    ):
        completion = model_rewrite_client.chat.completions.create(
            model=MODEL_REWRITE_CLIENT_MODEL,
            messages=[{"role": "user", "content": "Say hi."}],
            max_tokens=16,
        )

        assert completion.model == VLLM_MODEL, (
            f"backend should report the rewritten model, got {completion.model!r}"
        )
        assert completion.choices, "backend should return at least one choice"

    def test_chat_completions_streaming_alias_rewritten_for_backend(
        self, model_rewrite_client
    ):
        stream = model_rewrite_client.chat.completions.create(
            model=MODEL_REWRITE_CLIENT_MODEL,
            messages=[{"role": "user", "content": "Say hi."}],
            max_tokens=16,
            stream=True,
        )

        models = set()
        chunks = 0
        for chunk in stream:
            chunks += 1
            if chunk.model:
                models.add(chunk.model)

        assert chunks, "streamed chat completion should yield at least one chunk"
        assert models == {VLLM_MODEL}, (
            f"every chunk should report the rewritten model, got {models!r}"
        )

    def test_chat_completions_default_model_injected(self, model_rewrite_proxy):
        """A request with no `model` picks up the configured default_model.

        The SDK requires `model`, so this drives the raw HTTP endpoint.
        """
        response = httpx.post(
            f"http://127.0.0.1:{model_rewrite_proxy}/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Say hi."}],
                "max_tokens": 16,
            },
            timeout=300.0,
        )

        assert response.status_code == 200, response.text
        assert response.json()["model"] == VLLM_MODEL, response.text


if __name__ == "__main__":
    sys.exit(
        pytest.main(
            [__file__, "-v", "--tb=short", "-ra", "--durations=20"] + sys.argv[1:]
        )
    )
