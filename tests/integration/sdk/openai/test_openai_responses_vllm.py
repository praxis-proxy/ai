#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "httpx>=0.27",
#     "openai>=2.0",
#     "pytest>=8.0",
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
from urllib.parse import urlparse

import pytest
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen3-0.6B")
OGX_BASE_URL = os.environ.get("OGX_BASE_URL", "http://127.0.0.1:8321")
PRAXIS_AI_BIN = os.environ.get("PRAXIS_AI_BIN")
CONFIG_PATH = "examples/configs/openai/responses/full-flow.yaml"
AGENTIC_CONFIG_PATH = "examples/configs/openai/responses/agentic-loop.yaml"

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
        raise FileNotFoundError(
            f"PRAXIS_AI_BIN={PRAXIS_AI_BIN!r} not found"
        )
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


def _write_config(praxis_port: int, db_path: str) -> str:
    with open(CONFIG_PATH) as f:
        config = f.read()

    config = config.replace("127.0.0.1:8080", f"127.0.0.1:{praxis_port}")
    config = config.replace("127.0.0.1:3001", _vllm_endpoint())
    config = config.replace("127.0.0.1:9999", _ogx_endpoint())
    config = config.replace(
        "sqlite://responses.db?mode=rwc",
        f"sqlite://{db_path}?mode=rwc",
    )

    fd, path = tempfile.mkstemp(suffix=".yaml")
    with os.fdopen(fd, "w") as f:
        f.write(config)
    return path


def _wait_for_proxy(port: int, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.2)
    raise TimeoutError(f"Praxis did not start within {timeout}s")


# ---------------------------------------------------------------------------
# MCP Mock Server
# ---------------------------------------------------------------------------


class MCPHandler(BaseHTTPRequestHandler):
    """Streamable HTTP MCP server with a single get_weather tool."""

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        req = json.loads(body)
        method = req.get("method")
        rid = req.get("id")

        if method == "initialize":
            self._json_rpc(rid, {
                "protocolVersion": "2025-03-26",
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "weather-mock", "version": "0.1.0"},
            })
        elif method == "notifications/initialized":
            self.send_response(202)
            self.end_headers()
        elif method == "tools/list":
            self._json_rpc(rid, {
                "tools": [{
                    "name": "get_weather",
                    "description": "Get current weather for a city",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                        "additionalProperties": False,
                    },
                }]
            })
        elif method == "tools/call":
            city = (
                req.get("params", {})
                .get("arguments", {})
                .get("city", "unknown")
            )
            self._json_rpc(rid, {
                "content": [
                    {"type": "text", "text": f"72F and sunny in {city}"}
                ]
            })
        elif method == "ping":
            self._json_rpc(rid, {})
        else:
            self._json_rpc(
                rid, None,
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

    def do_GET(self):
        payload = json.dumps({
            "web": {
                "results": [
                    {
                        "title": "Mock Search Result",
                        "url": "https://example.com/mock",
                        "description": "A mock search result for testing",
                    }
                ]
            }
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt, *args):
        pass


def _write_agentic_config(
    praxis_port: int, db_path: str, mcp_port: int,
    search_port: int,
) -> str:
    """Patch agentic-loop.yaml with test ports and allow_loopback."""
    with open(AGENTIC_CONFIG_PATH) as f:
        config = f.read()

    config = config.replace("127.0.0.1:8080", f"127.0.0.1:{praxis_port}")
    vllm = _vllm_endpoint()
    config = config.replace(
        '- "127.0.0.1:3001"',
        f'- "{vllm}"\n'
        "                    read_timeout_ms: 120000",
    )
    config = config.replace(
        "sqlite://responses.db?mode=rwc",
        f"sqlite://{db_path}?mode=rwc",
    )
    config = config.replace(
        "- filter: openai_mcp_tool_resolve\n",
        "- filter: openai_mcp_tool_resolve\n"
        "        allow_loopback: true\n",
    )
    config = config.replace(
        "- filter: openai_mcp_dispatch\n"
        "              - filter: agentic_loop",
        "- filter: openai_mcp_dispatch\n"
        "                allow_loopback: true\n"
        "              - filter: agentic_loop",
    )
    config = config.replace(
        "max_iterations: 11\n",
        "max_iterations: 11\n"
        "        timeout_ms: 120000\n"
        "        step_timeout_ms: 120000\n",
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
        _wait_for_proxy(port)
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
def openai_client(praxis_proxy):
    """Return an OpenAI client pointed at the local Praxis proxy."""
    return OpenAI(
        base_url=f"http://127.0.0.1:{praxis_proxy}/v1",
        api_key="test",
        max_retries=0,
        timeout=180,
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
            store=False,
            max_output_tokens=128,
        )

        assert response.status == "completed"
        assert "HELLO-PRAXIS" in response.output_text

    def test_store_and_retrieve(self, openai_client):
        response = openai_client.responses.create(
            model=VLLM_MODEL,
            input="Say exactly: STORED-OK /no_think",
            store=True,
            max_output_tokens=128,
        )

        assert response.status == "completed"
        assert response.id

        retrieved = openai_client.responses.retrieve(response.id)

        assert retrieved.id == response.id
        assert retrieved.status == "completed"

    def test_rehydrated_second_turn(self, openai_client):
        first = openai_client.responses.create(
            model=VLLM_MODEL,
            input=(
                "Remember this nonce: VIOLET-7319. "
                "Acknowledge it. /no_think"
            ),
            store=True,
            max_output_tokens=128,
        )

        assert first.status == "completed"

        second = openai_client.responses.create(
            model=VLLM_MODEL,
            input=(
                "What nonce did I just tell you? "
                "Repeat it exactly. /no_think"
            ),
            previous_response_id=first.id,
            store=True,
            max_output_tokens=128,
        )

        assert second.status == "completed"
        assert "VIOLET-7319" in second.output_text

    def test_doc_extract_inline_file_to_input_text(self, openai_client):
        """Issue #397: inline file_data is extracted to input_text and
        consumed by vLLM inference.

        Sends an input_file with base64-encoded text through the full
        pipeline (file_resolve → doc_extract → responses_proxy → vLLM).
        The doc_extract filter converts the input_file to input_text
        before forwarding. Asserts a unique marker from the document
        appears in the model output, proving vLLM consumed the
        extracted text.
        """
        import base64

        marker = "PRAXIS-DOC-9271"
        file_content = f"The secret marker is: {marker}"
        file_data = (
            "data:text/plain;base64,"
            + base64.b64encode(file_content.encode()).decode()
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
            store=False,
            max_output_tokens=128,
        )

        assert response.status == "completed"
        assert marker in response.output_text, (
            f"vLLM should produce output containing the document "
            f"marker '{marker}'; got: {response.output_text}"
        )

    def test_file_id_resolution_through_ogx(self, openai_client):
        """End-to-end: upload to OGX via Praxis, reference by file_id,
        verify vLLM output contains the file content.

        Pipeline: file_resolve (OGX) -> doc_extract -> responses_proxy -> vLLM
        """
        import io

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

    def test_client_function_call_returns_to_client(self, openai_client):
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
            store=False,
            max_output_tokens=256,
        )

        assert response.status == "completed"

        function_calls = [
            item for item in response.output
            if item.type == "function_call"
        ]
        assert len(function_calls) >= 1, (
            "vLLM should return at least one function_call; "
            f"got output types: {[i.type for i in response.output]}"
        )
        fc = function_calls[0]
        assert fc.name == "get_weather"
        args = json.loads(fc.arguments)
        assert "city" in args, f"function arguments should contain city: {args}"

    def test_streaming(self, openai_client):
        stream = openai_client.responses.create(
            model=VLLM_MODEL,
            input="Say exactly: STREAM-OK /no_think",
            store=False,
            stream=True,
            max_output_tokens=128,
        )

        event_types = []
        text_parts = []
        final_status = None

        for event in stream:
            event_types.append(event.type)
            if event.type == "response.output_text.delta":
                text_parts.append(event.delta)
            if event.type == "response.completed":
                final_status = event.response.status

        assert event_types[0] == "response.created"
        assert event_types[-1] == "response.completed"
        assert final_status == "completed"
        assert "STREAM-OK" in "".join(text_parts)


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
    """Start a Praxis proxy with the agentic-loop config."""
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
        _wait_for_proxy(port)
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
        timeout=180,
    )


# ---------------------------------------------------------------------------
# Agentic Loop Tests
# ---------------------------------------------------------------------------


class TestAgenticLoopVLLM:
    """Integration tests for the agentic loop against a vLLM backend."""

    def test_mcp_tool_auto_executes_and_returns_final_answer(
        self, agentic_client, agentic_proxy,
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
            1 for t in output_types
            if t in ("function_call", "message", "reasoning")
        )
        assert rounds >= 2, (
            "accumulated output should span at least two inference "
            f"rounds; got: {output_types}"
        )

    def test_client_function_exits_agentic_loop(self, agentic_client):
        """Client-side function tools exit the agentic loop without
        auto-execution, even when the IRR is active.

        This proves the IRR + agentic_loop + mcp_dispatch correctly
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
            store=False,
            max_output_tokens=256,
        )

        assert response.status == "completed"

        function_calls = [
            item for item in response.output
            if item.type == "function_call"
        ]
        assert len(function_calls) >= 1, (
            "client-side function calls should be returned to the caller; "
            f"got output types: {[i.type for i in response.output]}"
        )
        assert function_calls[0].name == "get_weather"


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
        timeout_ms: 120000
        step_timeout_ms: 60000
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
                callout_failure_mode: closed
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
                    endpoints:
                      - "{vllm_endpoint}"
            on_result:
              - filter: openai_file_search_callout
                key: pending
                value: "true"
                next: inference
              - default: true
                done: true
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
    embedding_dimension = int(
        os.environ.get("OGX_EMBEDDING_DIMENSION", "768")
    )

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
            status = client.get(
                f"/v1/vector_stores/{store_id}/files/{file_id}"
            ).json()
            if status.get("status") == "completed":
                break
            if status.get("status") in ("failed", "cancelled"):
                raise RuntimeError(
                    f"OGX indexing failed: {status.get('last_error')}"
                )
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
        _wait_for_proxy(port)
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
        timeout=180,
    )


class TestFileSearchVLLM:
    """File search integration tests: vLLM -> Praxis -> OGX -> vLLM."""

    def test_file_search_with_vllm_translation(
        self, file_search_client, vector_store
    ):
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
            item
            for item in response.output
            if item.type == "file_search_call"
        ]
        assert file_search_items, (
            "translated function_call(name=file_search) should appear as "
            f"file_search_call; got output types: {output_types}"
        )
        for item in file_search_items:
            assert item.status in ("completed", "incomplete"), (
                f"file_search_call status should be terminal; "
                f"got: {item.status}"
            )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"] + sys.argv[1:]))
